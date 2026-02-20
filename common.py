# region Imports
import random
from datetime import timedelta
import os
from argparse import Namespace
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

# Classifiers
from classifiers.differential.RAT import RAT, RAT_baseline, RAT_selfattn, RAT_zeroref
from classifiers.FFBase import FFBase
from classifiers.single_input.FF import FF

# Config
from config import local_config

# Datasets
from datasets.ASVspoof5 import ASVspoof5Dataset_pair, ASVspoof5Dataset_single
from datasets.ASVspoof2019 import ASVspoof2019LADataset_pair, ASVspoof2019LADataset_single
from datasets.ASVspoof2021 import (
    ASVspoof2021DFDataset_pair,
    ASVspoof2021DFDataset_single,
    ASVspoof2021LADataset_pair,
    ASVspoof2021LADataset_single,
)
from datasets.InTheWild import InTheWildDataset_pair, InTheWildDataset_single
from datasets.Morphing import MorphingDataset_pair, MorphingDataset_single
from datasets.utils import custom_pair_batch_create, custom_single_batch_create

# Extractors
from extractors.HuBERT import HuBERT_base, HuBERT_extralarge, HuBERT_large
from extractors.Wav2Vec2 import Wav2Vec2_base, Wav2Vec2_large, Wav2Vec2_LV60k
from extractors.WavLM import WavLM_base, WavLM_baseplus, WavLM_large
from extractors.XLSR import XLSR_1B, XLSR_2B, XLSR_300M

# Feature processors
from feature_processors.MeanProcessor import MeanProcessor

# Trainers
from trainers.BaseTrainer import BaseTrainer
from trainers.FFPairTrainer import FFPairTrainer
from trainers.FFTrainer import FFTrainer

# endregion

# map of argument names to the classes
EXTRACTORS: dict[str, type] = {
    "HuBERT_base": HuBERT_base,
    "HuBERT_large": HuBERT_large,
    "HuBERT_extralarge": HuBERT_extralarge,
    "Wav2Vec2_base": Wav2Vec2_base,
    "Wav2Vec2_large": Wav2Vec2_large,
    "Wav2Vec2_LV60k": Wav2Vec2_LV60k,
    "WavLM_base": WavLM_base,
    "WavLM_baseplus": WavLM_baseplus,
    "WavLM_large": WavLM_large,
    "XLSR_300M": XLSR_300M,
    "XLSR_1B": XLSR_1B,
    "XLSR_2B": XLSR_2B,
}
CLASSIFIERS: Dict[str, Tuple[type, Dict[str, type]]] = {
    # Maps the classifier to tuples of the corresponding class and the initializable arguments
    "FF": (FF, {}),
    "RAT": (RAT, {}),
    "RAT_baseline": (RAT_baseline, {}),
    "RAT_selfattn": (RAT_selfattn, {}),
    "RAT_zeroref": (RAT_zeroref, {}),
}
TRAINERS = {  # Maps the classifier to the trainer
    "FF": FFTrainer,
    "RAT": FFPairTrainer,
    "RAT_baseline": FFPairTrainer,
    "RAT_selfattn": FFPairTrainer,
    "RAT_zeroref": FFPairTrainer,
}
DATASETS = {  # map the dataset name to the dataset class
    "ASVspoof2019LADataset_single": ASVspoof2019LADataset_single,
    "ASVspoof2019LADataset_pair": ASVspoof2019LADataset_pair,
    "ASVspoof2021LADataset_single": ASVspoof2021LADataset_single,
    "ASVspoof2021LADataset_pair": ASVspoof2021LADataset_pair,
    "ASVspoof2021DFDataset_single": ASVspoof2021DFDataset_single,
    "ASVspoof2021DFDataset_pair": ASVspoof2021DFDataset_pair,
    "InTheWildDataset_single": InTheWildDataset_single,
    "InTheWildDataset_pair": InTheWildDataset_pair,
    "MorphingDataset_single": MorphingDataset_single,
    "MorphingDataset_pair": MorphingDataset_pair,
    "ASVspoof5Dataset_single": ASVspoof5Dataset_single,
    "ASVspoof5Dataset_pair": ASVspoof5Dataset_pair,
}


class DistributedWeightedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False, weights=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + self.rank)
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist()
        return iter(indices)


def get_dataloaders(
    dataset: str = "ASVspoof2019LADataset_pair",
    config: dict = local_config,
    lstm: bool = False,
    augment: bool = False,
    eval_only: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader] | DataLoader:

    # Get the dataset class and config
    # Always train on ASVspoof2019LA, evaluate on the specified dataset (except ASVspoof5)
    dataset_config = {}
    t = "pair" if "pair" in dataset else "single"
    if "ASVspoof2019LA" in dataset:
        train_dataset_class = DATASETS[dataset]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["asvspoof2019la"]
    elif "ASVspoof2021" in dataset:
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["asvspoof2021la"] if "LA" in dataset else config["asvspoof2021df"]
    elif "InTheWild" in dataset:
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["inthewild"]
    elif "Morphing" in dataset:
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["morphing"]
    elif "ASVspoof5" in dataset:
        train_dataset_class = DATASETS[dataset]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["asvspoof5"]
    # elif "MLDF" in dataset:
    #     return mldf_dataloader
    else:
        raise ValueError("Invalid dataset name.")

    # Common parameters
    collate_func = custom_single_batch_create if "single" in dataset else custom_pair_batch_create
    bs = config["batch_size"] if not lstm else config["lstm_batch_size"]  # Adjust batch size for LSTM models

    # Load the datasets
    train_dataloader = DataLoader(Dataset())  # dummy dataloader for type hinting compliance
    val_dataloader = DataLoader(Dataset())  # dummy dataloader for type hinting compliance
    if not eval_only:
        if rank == 0:
            print("Loading training dataset...")
        train_dataset = train_dataset_class(
            root_dir=config["data_dir"] + dataset_config["train_subdir"],
            protocol_file_name=dataset_config["train_protocol"],
            variant="train",
            augment=augment,
            rir_root=config["rir_root"],
        )

        dev_kwargs = {  # kwargs for the dataset class
            "root_dir": config["data_dir"] + dataset_config["dev_subdir"],
            "protocol_file_name": dataset_config["dev_protocol"],
            "variant": "dev",
        }
        if "2021DF" in dataset:  # 2021DF has a local variant
            dev_kwargs["local"] = True if "--local" in config["argv"] else False
            dev_kwargs["variant"] = "progress"
            val_dataset = eval_dataset_class(**dev_kwargs)
        else:
            # Create the dataset based on dynamically created dev_kwargs
            val_dataset = train_dataset_class(**dev_kwargs)

        # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
        # samples_weights = [train_dataset.get_class_weights()[i] for i in train_dataset.get_labels()]  # old and slow solution
        samples_weights = np.vectorize(train_dataset.get_class_weights().__getitem__)(
            train_dataset.get_labels()
        )  # blazing fast solution
        if distributed:
            sampler = DistributedWeightedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, weights=samples_weights, drop_last=True, seed=seed)
        else:
            sampler = WeightedRandomSampler(samples_weights, len(train_dataset))

        # create dataloader, use custom collate_fn to pad the data to the longest recording in batch
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=bs,
            collate_fn=collate_func,
            sampler=sampler,
            drop_last=True,
            num_workers=3 if distributed else 0,
            pin_memory=True,
            persistent_workers=True if distributed else False,
        )
        
        if distributed:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False, seed=seed)
        else:
            val_sampler = None

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=bs,
            collate_fn=collate_func,
            shuffle=False if distributed else True,
            sampler=val_sampler,
            num_workers=3 if distributed else 0,
            pin_memory=True,
            persistent_workers=True if distributed else False,
        )

    if rank == 0:
        print("Loading eval dataset...")
    eval_kwargs = {  # kwargs for the dataset class
        "root_dir": config["data_dir"] + dataset_config["eval_subdir"],
        "protocol_file_name": dataset_config["eval_protocol"],
        "variant": "eval",
    }
    if "2021DF" in dataset:  # 2021DF has a local variant
        eval_kwargs["local"] = True if "--local" in config["argv"] else False

    # Create the dataset based on dynamically created eval_kwargs
    eval_dataset = eval_dataset_class(**eval_kwargs)
    
    if distributed:
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False, seed=seed)
    else:
        eval_sampler = None
        
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=bs,
        collate_fn=collate_func,
        shuffle=False if distributed else True,
        sampler=eval_sampler,
        num_workers=3 if distributed else 0,
        pin_memory=True,
        persistent_workers=True if distributed else False,
    )

    if eval_only:
        return eval_dataloader
    else:
        return train_dataloader, val_dataloader, eval_dataloader


def build_model(args: Namespace, device: str = "cpu") -> Tuple[FFBase, BaseTrainer]:

    # region Extractor
    extractor = EXTRACTORS[args.extractor]()  # map the argument to the class and instantiate it
    # endregion

    # region Processor (pooling)
    processor = None
    if args.processor == "Mean":
        processor = MeanProcessor()  # default avg pooling along the transformer layers and time frames
    else:
        raise ValueError("Only Mean processor is currently supported.")
    # endregion

    # region Model and trainer
    try:
        cls_input_dim = extractor.feature_size
        model_cls, _ = CLASSIFIERS[str(args.classifier)]
        model = model_cls(
            extractor, processor, in_dim=cls_input_dim
        )
        trainer_cls = TRAINERS[str(args.classifier)]
        trainer = trainer_cls(model, device=device)
    except KeyError:
        raise ValueError(f"Invalid classifier: {args.classifier}. Should be one of: {list(CLASSIFIERS.keys())}")
    # endregion

    # Print model info
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Building {type(model).__name__} model with {type(model.extractor).__name__} extractor", end="")
        if isinstance(model, FFBase) or hasattr(model, "feature_processor"):
             print(f" and {type(model.feature_processor).__name__} processor.")
        else:
             print(".")

    return model, trainer


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
