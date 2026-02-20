#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader

from config import local_config
from common import build_model, get_dataloaders, setup_ddp, cleanup_ddp, set_seed
from parse_arguments import parse_args
from trainers import FFPairTrainer


def main():
    args = parse_args()

    set_seed(args.seed)

    is_ddp, rank, local_rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    config = local_config

    model, trainer = build_model(args, device=device)

    if rank == 0:
        print(f"Trainer: {type(trainer).__name__}")

    # Load the model from the checkpoint
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
    else:
        raise ValueError("Checkpoint must be specified when only evaluating.")

    # Load the dataset
    train_dataloader, dev_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        eval_only=False,
        distributed=is_ddp,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    assert isinstance( # Is here for type checking and hinting compliance
        eval_dataloader, DataLoader
    ), "Error type of eval_dataloader returned from get_dataloaders."

    if rank == 0:
        print(
            f"Evaluating {args.checkpoint} {type(model).__name__} on "
            + f"{type(eval_dataloader.dataset).__name__} dataloader."
        )

    trainer.eval(eval_dataloader)

    try:
        del eval_dataloader
        cleanup_ddp()
    except OSError as e:
        if "[Errno 16] Device or resource busy" in str(e):
            print("DDP cleanup: Device or resource busy error, ignoring.")
        else:
            raise e


if __name__ == "__main__":
    main()
