###########################################################################################
# Modified training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs, Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

"""

MACE utils for use in psiflow -- copied from mace@dee204f
The following changes were made:

    - use signal module to wrap tools.train() call with timeout such that
    there is time left to save the best model.

    - build model from scratch but load state dict of starting model

    - simplified Calculator which incorporates additional atomic energy offsets

"""
import argparse


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def run(rank: int, args: argparse.Namespace, world_size: int) -> None:
    import ast
    import glob
    import json
    import logging
    import os
    from copy import deepcopy
    from pathlib import Path
    from typing import Optional, List

    import mace
    import numpy as np
    import torch
    import torch.nn.functional
    from torch.optim import LBFGS
    from torch.utils.data import ConcatDataset
    from e3nn import o3
    from e3nn.util import jit
    from mace import data, modules, tools
    from mace.data import KeySpecification, update_keyspec_from_kwargs
    from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq
    from mace.cli.convert_oeq_e3nn import run as run_oeq_to_e3nn
    from mace.cli.visualise_train import TrainingPlotter
    from mace.tools import torch_geometric
    from mace.tools.model_script_utils import configure_model
    from mace.tools.scripts_utils import (
        LRScheduler,
        get_atomic_energies,
        get_config_type_weights,
        get_dataset_from_xyz,
        check_path_ase_read,
        SubsetCollection,
        dict_to_array,
        get_loss_fn,
        get_avg_num_neighbors,
        get_params_options,
        get_optimizer,
        get_swa,
        setup_wandb,
        print_git_commit,
        extract_config_mace_model,
        convert_to_json_format,
        get_files_with_suffix,
    )
    from mace.tools.multihead_tools import (
        prepare_default_head,
        dict_head_to_dataclass,
        HeadConfig,
    )
    from mace.tools.utils import AtomicNumberTable
    from mace.tools.run_train_utils import (
        normalize_file_paths, 
        load_dataset_for_path,
        combine_datasets,
    )
    from mace.tools.tables_utils import create_error_table
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.optim.swa_utils import SWALR, AveragedModel
    from torch_ema import ExponentialMovingAverage

    # extend MACE arg parser with ability to pass initialized model; set tmpdirs
    args.log_dir = os.path.join(os.getcwd(), "log")
    args.model_dir = os.path.join(os.getcwd())
    args.results_dir = os.path.join(os.getcwd(), "results")
    args.downloads_dir = os.path.join(os.getcwd(), "downloads")
    args.checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")

    tag = tools.get_tag(name=args.name, seed=args.seed)
    args, input_log_messages = tools.check_args(args)

    # default keyspec to update using heads dictionary
    args.key_specification = KeySpecification()
    update_keyspec_from_kwargs(args.key_specification, vars(args))

    if args.distributed:
        local_rank = rank
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)
    else:
        pass

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=rank)
    logging.info("===========VERIFYING SETTINGS===========")
    for message, loglevel in input_log_messages:
        logging.log(level=loglevel, msg=message)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.debug(f"Configuration: {args}")
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)
    commit = print_git_commit()
    model_foundation: Optional[torch.nn.Module] = None

    assert args.foundation_model is None
    assert args.statistics_file is None

    # setup default head
    args.heads = prepare_default_head(args)

    logging.info("===========LOADING INPUT DATA===========")
    heads = list(args.heads.keys())
    logging.info(f"Using heads: {heads}")
    logging.info("Using the key specifications to parse data:")
    for name, head_dict in args.heads.items():
        head_keyspec = head_dict["key_specification"]
        logging.info(f"{name}: {head_keyspec}")

    head_configs: List[HeadConfig] = []
    for head, head_args in args.heads.items():
        logging.info(f"=============    Processing head {head}     ===========")
        head_config = dict_head_to_dataclass(head_args, head, args)

        # Handle train_file and valid_file - normalize to lists
        if hasattr(head_config, "train_file") and head_config.train_file is not None:
            head_config.train_file = normalize_file_paths(head_config.train_file)
        if hasattr(head_config, "valid_file") and head_config.valid_file is not None:
            head_config.valid_file = normalize_file_paths(head_config.valid_file)
        if hasattr(head_config, "test_file") and head_config.test_file is not None:
            head_config.test_file = normalize_file_paths(head_config.test_file)

        
        train_files_ase_list = [
            f for f in head_config.train_file if check_path_ase_read(f)
        ]
        valid_files_ase_list = None
        test_files_ase_list = None
        if head_config.valid_file:
            valid_files_ase_list = [
                f for f in head_config.valid_file if check_path_ase_read(f)
            ]
        if head_config.test_file:
            test_files_ase_list = [
                f for f in head_config.test_file if check_path_ase_read(f)
            ]
        config_type_weights = get_config_type_weights(
            head_config.config_type_weights
        )
        collections, atomic_energies_dict = get_dataset_from_xyz(
            work_dir=args.model_dir,
            train_path=train_files_ase_list,
            valid_path=valid_files_ase_list,
            valid_fraction=head_config.valid_fraction,
            config_type_weights=config_type_weights,
            test_path=test_files_ase_list,
            seed=args.seed,
            key_specification=head_config.key_specification,
            head_name=head_config.head_name,
            keep_isolated_atoms=head_config.keep_isolated_atoms,
        )
        head_config.collections = SubsetCollection(
            train=collections.train,
            valid=collections.valid,
            tests=collections.tests,
        )
        head_config.atomic_energies_dict = atomic_energies_dict
        logging.info(
            f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
            f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}],"
        )
        head_configs.append(head_config)

    if all(
        check_path_ase_read(head_config.train_file[0]) for head_config in head_configs
    ):
        size_collections_train = sum(
            len(head_config.collections.train) for head_config in head_configs
        )
        size_collections_valid = sum(
            len(head_config.collections.valid) for head_config in head_configs
        )
        if size_collections_train < args.batch_size:
            logging.error(
                f"Batch size ({args.batch_size}) is larger than the number of training data ({size_collections_train})"
            )
        if size_collections_valid < args.valid_batch_size:
            logging.warning(
                f"Validation batch size ({args.valid_batch_size}) is larger than the number of validation data ({size_collections_valid})"
            )

    # Atomic number table
    # yapf: disable
    for head_config in head_configs:
        if head_config.atomic_numbers is None:
            assert all(check_path_ase_read(f) for f in head_config.train_file), "Must specify atomic_numbers when using .h5 or .aselmdb train_file input"
            z_table_head = tools.get_atomic_number_table_from_zs(
                z
                for configs in (head_config.collections.train, head_config.collections.valid)
                for config in configs
                for z in config.atomic_numbers
            )
            head_config.atomic_numbers = z_table_head.zs
            head_config.z_table = z_table_head
        else:
            if head_config.statistics_file is None:
                logging.info("Using atomic numbers from command line argument")
            else:
                logging.info("Using atomic numbers from statistics file")
            zs_list = ast.literal_eval(head_config.atomic_numbers)
            assert isinstance(zs_list, list)
            z_table_head = tools.AtomicNumberTable(zs_list)
            head_config.atomic_numbers = zs_list
            head_config.z_table = z_table_head
        # yapf: enable
    all_atomic_numbers = set()
    for head_config in head_configs:
        all_atomic_numbers.update(head_config.atomic_numbers)
    z_table = AtomicNumberTable(sorted(list(all_atomic_numbers)))
    logging.info(f"Atomic Numbers used: {z_table.zs}")

    # Atomic energies
    atomic_energies_dict = {}
    for head_config in head_configs:
        if head_config.atomic_energies_dict is None or len(head_config.atomic_energies_dict) == 0:
            assert head_config.E0s is not None, "Atomic energies must be provided"
            if all(check_path_ase_read(f) for f in head_config.train_file) and head_config.E0s.lower() != "foundation":
                atomic_energies_dict[head_config.head_name] = get_atomic_energies(
                    head_config.E0s, head_config.collections.train, head_config.z_table
                )
            elif head_config.E0s.lower() == "foundation":
                raise NotImplementedError
            else:
                atomic_energies_dict[head_config.head_name] = get_atomic_energies(head_config.E0s, None, head_config.z_table)
        else:
            atomic_energies_dict[head_config.head_name] = head_config.atomic_energies_dict

    heads = sorted(heads, key=lambda x: -1000 if x == "pt_head" else 0)

    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        if args.E0s.lower() == "foundation":
            raise NotImplementedError
        else:
            if args.train_file.endswith(".xyz"):
                atomic_energies_dict = get_atomic_energies(
                    args.E0s, collections.train, z_table
                )
            else:
                atomic_energies_dict = get_atomic_energies(args.E0s, None, z_table)

    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        args.compute_dipole = True
        args.compute_energy = False
        args.compute_forces = False
        args.compute_virials = False
        args.compute_stress = False
        args.compute_polarizability = False
    elif args.model == "AtomicDielectricMACE":
        atomic_energies = None
        dipole_only = False
        args.compute_dipole = True
        args.compute_energy = False
        args.compute_forces = False
        args.compute_virials = False
        args.compute_stress = False
        args.compute_polarizability = True
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            args.compute_dipole = True
            args.compute_energy = True
            args.compute_forces = True
            args.compute_virials = False
            args.compute_stress = False
            args.compute_polarizability = False
        else:
            args.compute_energy = True
            args.compute_dipole = False
            args.compute_polarizability = False
        # atomic_energies: np.ndarray = np.array(
        #     [atomic_energies_dict[z] for z in z_table.zs]
        # )
        atomic_energies = dict_to_array(atomic_energies_dict, heads)
        for head_config in head_configs:
            try:
                logging.info(f"Atomic Energies used (z: eV) for head {head_config.head_name}: " + "{" + ", ".join([f"{z}: {atomic_energies_dict[head_config.head_name][z]}" for z in head_config.z_table.zs]) + "}")
            except KeyError as e:
                raise KeyError(f"Atomic number {e} not found in atomic_energies_dict for head {head_config.head_name}, add E0s for this atomic number") from e

    args.batch_size = min(len(collections.train), args.batch_size)
    print("actual batch size: {}".format(args.batch_size))

    # Load datasets for each head, supporting multiple files per head
    valid_sets = {head: [] for head in heads}
    train_sets = {head: [] for head in heads}

    for head_config in head_configs:
        train_datasets = []

        logging.info(f"Processing datasets for head '{head_config.head_name}'")

        ase_files = [f for f in head_config.train_file if check_path_ase_read(f)]

        if ase_files:
            dataset = load_dataset_for_path(
            file_path=ase_files,
            r_max=args.r_max,
            z_table=z_table,
            head_config=head_config,
            heads=heads,
            collection=head_config.collections.train,
            )
            train_datasets.append(dataset)
            logging.debug(f"Successfully loaded dataset from ASE files: {ase_files}")

        if not train_datasets:
            raise ValueError(f"No valid training datasets found for head {head_config.head_name}")

        train_sets[head_config.head_name] = combine_datasets(train_datasets, head_config.head_name)

        if head_config.valid_file:
            valid_datasets = []

            valid_ase_files = [f for f in head_config.valid_file if check_path_ase_read(f)]
            valid_non_ase_files = [f for f in head_config.valid_file if not check_path_ase_read(f)]

            if valid_ase_files:
                valid_dataset = load_dataset_for_path(
                    file_path=valid_ase_files,
                    r_max=args.r_max,
                    z_table=z_table,
                    head_config=head_config,
                    heads=heads,
                    collection=head_config.collections.valid,
                )
                valid_datasets.append(valid_dataset)
                logging.debug(f"Successfully loaded validation dataset from ASE files: {valid_ase_files}")
            for valid_file in valid_non_ase_files:
                valid_dataset = load_dataset_for_path(
                file_path=valid_file,
                r_max=args.r_max,
                z_table=z_table,
                head_config=head_config,
                heads=heads,
            )
                valid_datasets.append(valid_dataset)
                logging.debug(f"Successfully loaded validation dataset from {valid_file}")

            # Combine validation datasets
            if valid_datasets:
                valid_sets[head_config.head_name] = combine_datasets(valid_datasets, f"{head_config.head_name}_valid")
                logging.info(f"Combined validation datasets for {head_config.head_name}")

        # If no valid file is provided but collection exist, use the validation set from the collection
        if head_config.valid_file is None and head_config.collections.valid:
            valid_sets[head_config.head_name] = [
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=args.r_max, heads=heads
                )
                for config in head_config.collections.valid
            ]
        if not valid_sets[head_config.head_name]:
            raise ValueError(f"No valid datasets found for head {head_config.head_name}, please provide a valid_file or a valid_fraction")

        # Create data loader for this head
        if isinstance(train_sets[head_config.head_name], list):
            dataset_size = len(train_sets[head_config.head_name])
        else:
            dataset_size = len(train_sets[head_config.head_name])
        logging.info(f"Head '{head_config.head_name}' training dataset size: {dataset_size}")

        train_loader_head = torch_geometric.dataloader.DataLoader(
            dataset=train_sets[head_config.head_name],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=(not args.lbfgs),
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        head_config.train_loader = train_loader_head


    # concatenate all the trainsets
    train_set = ConcatDataset([train_sets[head] for head in heads])
    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=(not args.lbfgs),
            seed=args.seed,
        )
        valid_samplers = {}
        for head, valid_set in valid_sets.items():
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
            valid_samplers[head] = valid_sampler

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=(train_sampler is None and not args.lbfgs),
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )

    valid_loaders = {heads[i]: None for i in range(len(heads))}
    if not isinstance(valid_sets, dict):
        valid_sets = {"Default": valid_sets}
    for head, valid_set in valid_sets.items():
        valid_loaders[head] = torch_geometric.dataloader.DataLoader(
            dataset=valid_set,
            batch_size=args.valid_batch_size,
            sampler=valid_samplers[head] if args.distributed else None,
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )

    loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole)
    args.avg_num_neighbors = get_avg_num_neighbors(head_configs, args, train_loader, device)
    logging.info(f"Average number of neighbors: {args.avg_num_neighbors}")

    # Model
    model, output_args = configure_model(args, train_loader, atomic_energies, model_foundation, heads, z_table, head_configs)
    model.to(device)

    logging.debug(model)
    logging.info(f"Total number of parameters: {tools.count_parameters(model)}")
    logging.info("")
    logging.info("===========OPTIMIZER INFORMATION===========")
    logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
    logging.info(f"Batch size: {args.batch_size}")
    if args.ema:
        logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
    logging.info(
        f"Number of gradient updates: {int(args.max_num_epochs*len(train_set)/args.batch_size)}"
    )
    logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    logging.info(loss_fn)

    # Cueq and OEQ conversion
    if args.enable_cueq and args.enable_oeq:
        logging.warning(
            "Both CUEQ and OEQ are enabled, using CUEQ for training. "
            "To use OEQ, disable CUEQ with --disable_cueq."
        )
        args.enable_oeq = False
    if args.enable_cueq and not args.only_cueq:
        logging.info("Converting model to CUEQ for accelerated training")
        assert model.__class__.__name__ in ["MACE", "ScaleShiftMACE", "MACELES"]
        model = run_e3nn_to_cueq(deepcopy(model), device=device)
    if args.enable_oeq:
        logging.info("Converting model to OEQ for accelerated training")
        assert model.__class__.__name__ in ["MACE", "ScaleShiftMACE", "MACELES"]
        model = run_e3nn_to_oeq(deepcopy(model), device=device)

    # Optimizer
    param_options = get_params_options(args, model)
    optimizer: torch.optim.Optimizer
    optimizer = get_optimizer(args, param_options)
    logger = tools.MetricsLogger(
        directory=args.results_dir, tag=tag + "_train"
    )  # pylint: disable=E1123

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        swa, swas = get_swa(args, model, optimizer, swas, dipole_only)

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    restart_lbfgs = False
    opt_start_epoch = None
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception:  # pylint: disable=W0703
            try:
                opt_start_epoch = checkpoint_handler.load_latest(
                    state=tools.CheckpointState(model, optimizer, lr_scheduler),
                    swa=False,
                    device=device,
                )
            except Exception: # pylint: disable=W0703
                restart_lbfgs = True
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    if args.lbfgs:
        logging.info("Switching optimizer to LBFGS")
        optimizer = LBFGS(model.parameters(),
                          history_size=200,
                          max_iter=20,
                          line_search_fn="strong_wolfe")
        if restart_lbfgs:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
            if opt_start_epoch is not None:
                start_epoch = opt_start_epoch

    if args.wandb:
        setup_wandb(args)
    if args.distributed:
        distributed_model = DDP(model, device_ids=[local_rank])
    else:
        distributed_model = None


    train_valid_data_loader = {}
    for head_config in head_configs:
        data_loader_name = "train_" + head_config.head_name
        train_valid_data_loader[data_loader_name] = head_config.train_loader
    for head, valid_loader in valid_loaders.items():
        data_load_name = "valid_" + head
        train_valid_data_loader[data_load_name] = valid_loader

    if args.plot and args.plot_frequency > 0:
        try:
            plotter = TrainingPlotter(
                results_dir=logger.path,
                heads=heads,
                table_type=args.error_table,
                train_valid_data=train_valid_data_loader,
                test_data={},
                output_args=output_args,
                device=device,
                plot_frequency=args.plot_frequency,
                distributed=args.distributed,
                swa_start=swa.start if swa else None,
                plot_interaction_e=args.plot_interaction_e
                )
        except Exception as e:  # pylint: disable=W0718
            logging.debug(f"Creating Plotter failed: {e}")
    else:
        plotter = None

    try:
        tools.train(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            valid_loaders=valid_loaders,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_handler=checkpoint_handler,
            eval_interval=args.eval_interval,
            start_epoch=start_epoch,
            max_num_epochs=args.max_num_epochs,
            logger=logger,
            patience=args.patience,
            save_all_checkpoints=args.save_all_checkpoints,
            output_args=output_args,
            device=device,
            swa=swa,
            ema=ema,
            max_grad_norm=args.clip_grad,
            log_errors=args.error_table,
            log_wandb=args.wandb,
            distributed=args.distributed,
            distributed_model=distributed_model,
            plotter=plotter,
            train_sampler=train_sampler,
            rank=rank,
        )
    except TimeoutException:
        logging.info("received SIGTERM!")
        pass

    logging.info("")
    logging.info("===========RESULTS===========")

    train_valid_data_loader = {}
    for head_config in head_configs:
        data_loader_name = "train_" + head_config.head_name
        train_valid_data_loader[data_loader_name] = head_config.train_loader
    for head, valid_loader in valid_loaders.items():
        data_load_name = "valid_" + head
        train_valid_data_loader[data_load_name] = valid_loader
    test_sets = {}
    stop_first_test = False
    test_data_loader = {}
    if all(
        head_config.test_file == head_configs[0].test_file
        for head_config in head_configs
    ) and head_configs[0].test_file is not None:
        stop_first_test = True
    if all(
        head_config.test_dir == head_configs[0].test_dir
        for head_config in head_configs
    ) and head_configs[0].test_dir is not None:
        stop_first_test = True
    for head_config in head_configs:
        if all(check_path_ase_read(f) for f in head_config.train_file):
            for name, subset in head_config.collections.tests:
                test_sets[name] = [
                    data.AtomicData.from_config(
                        config, z_table=z_table, cutoff=args.r_max, heads=heads
                    )
                    for config in subset
                ]
        if head_config.test_dir is not None:
            if not args.multi_processed_test:
                test_files = get_files_with_suffix(head_config.test_dir, "_test.h5")
                for test_file in test_files:
                    name = os.path.splitext(os.path.basename(test_file))[0]
                    test_sets[name] = data.HDF5Dataset(
                        test_file, r_max=args.r_max, z_table=z_table, heads=heads, head=head_config.head_name
                    )
            else:
                test_folders = glob(head_config.test_dir + "/*")
                for folder in test_folders:
                    name = os.path.splitext(os.path.basename(test_file))[0]
                    test_sets[name] = data.dataset_from_sharded_hdf5(
                        folder, r_max=args.r_max, z_table=z_table, heads=heads, head=head_config.head_name
                    )
        for test_name, test_set in test_sets.items():
            test_sampler = None
            if args.distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_set,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                    seed=args.seed,
                )
            try:
                drop_last = test_set.drop_last
            except AttributeError as e:  # pylint: disable=W0612
                drop_last = False
            test_loader = torch_geometric.dataloader.DataLoader(
                test_set,
                batch_size=args.valid_batch_size,
                shuffle=(test_sampler is None),
                drop_last=drop_last,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            test_data_loader[test_name] = test_loader
        if stop_first_test:
            break


    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        model.to(device)
        if args.distributed:
            distributed_model = DDP(model, device_ids=[local_rank])
        model_to_evaluate = model if not args.distributed else distributed_model
        if swa_eval:
            logging.info(f"Loaded Stage two model from epoch {epoch} for evaluation")
        else:
            logging.info(f"Loaded Stage one model from epoch {epoch} for evaluation")

        if rank == 0:
            # Save entire model
            if swa_eval:
                model_path = Path(args.checkpoints_dir) / (tag + "_stagetwo.model")
            else:
                model_path = Path(args.checkpoints_dir) / (tag + ".model")
            logging.info(f"Saving model to {model_path}")
            model_to_save = deepcopy(model)
            if args.enable_cueq and not args.only_cueq:
                logging.info("RUNING CUEQ TO E3NN")
                model_to_save = run_cueq_to_e3nn(deepcopy(model), device=device)
            if args.enable_oeq:
                logging.info("RUNING OEQ TO E3NN")
                model_to_save = run_oeq_to_e3nn(deepcopy(model), device=device)
            if args.save_cpu:
                model_to_save = model_to_save.to("cpu")
            torch.save(model_to_save, model_path)
            extra_files = {
                "commit.txt": commit.encode("utf-8") if commit is not None else b"",
                "config.yaml": json.dumps(
                    convert_to_json_format(extract_config_mace_model(model))
                ),
            }
            if swa_eval:
                torch.save(
                    model_to_save, Path(args.model_dir) / (args.name + "_stagetwo.model")
                )
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_stagetwo_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0718
                    pass
            else:
                torch.save(model_to_save, Path(args.model_dir) / (args.name + ".model"))
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata to {path_complied}")
                    model_compiled = jit.compile(deepcopy(model_to_save))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0718
                    pass

        logging.info("Computing metrics for training, validation, and test sets")
        for param in model.parameters():
            param.requires_grad = False
        skip_heads = args.skip_evaluate_heads.split(",") if args.skip_evaluate_heads else []
        if skip_heads:
            logging.info(f"Skipping evaluation for heads: {skip_heads}")
        table_train_valid = create_error_table(
            table_type=args.error_table,
            all_data_loaders=train_valid_data_loader,
            model=model_to_evaluate,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
            distributed=args.distributed,
            skip_heads=skip_heads,
        )
        logging.info("Error-table on TRAIN and VALID:\n" + str(table_train_valid))

        if test_data_loader:
            table_test = create_error_table(
                table_type=args.error_table,
                all_data_loaders=test_data_loader,
                model=model_to_evaluate,
                loss_fn=loss_fn,
                output_args=output_args,
                log_wandb=args.wandb,
                device=device,
                distributed=args.distributed,
            )
            logging.info("Error-table on TEST:\n" + str(table_test))
        if args.plot:
            try:
                plotter = TrainingPlotter(
                    results_dir=logger.path,
                    heads=heads,
                    table_type=args.error_table,
                    train_valid_data=train_valid_data_loader,
                    test_data=test_data_loader,
                    output_args=output_args,
                    device=device,
                    plot_frequency=args.plot_frequency,
                    distributed=args.distributed,
                    swa_start=swa.start if swa else None
                )
                plotter.plot(epoch, model_to_evaluate, rank)
            except Exception as e:  # pylint: disable=W0718
                logging.debug(f"Plotting failed: {e}")

        if args.distributed:
            torch.distributed.barrier()

    logging.info("Done")
    if args.distributed:
        torch.distributed.destroy_process_group()


def main():
    import signal

    import torch
    from mace import tools

    signal.signal(signal.SIGTERM, timeout_handler)
    # main()
    parser = tools.build_default_arg_parser()
    parser.add_argument(
        "--initialized_model",
        help="path to initialized model",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        args.distributed = True
        import torch.multiprocessing as mp

        mp.spawn(run, args=(args, world_size), nprocs=world_size)
    else:
        args.distributed = False
        run(0, args, 1)
