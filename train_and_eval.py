#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
from common import build_model, get_dataloaders, setup_ddp, cleanup_ddp, set_seed
from config import local_config
from parse_arguments import parse_args

# trainers
from trainers.BaseFFTrainer import BaseFFTrainer


def main():
    args = parse_args()
    
    set_seed(args.seed)

    is_ddp, rank, local_rank, world_size = setup_ddp()
    
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    config = local_config

    model, trainer = build_model(args, device=device)

    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        augment=args.augment,
        distributed=is_ddp,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )

    if rank == 0:
        print(f"Training on {type(train_dataloader.dataset).__name__} dataloader.")

    # Train the model
    if isinstance(trainer, BaseFFTrainer):
        # Default value of numepochs = 20
        trainer.train(train_dataloader, val_dataloader, numepochs=args.num_epochs)
        trainer.eval(eval_dataloader, subtitle=str(args.num_epochs))  # Eval after training
    else:
        raise ValueError("Invalid trainer, should inherit from BaseFFTrainer.")

    try:
        del train_dataloader, val_dataloader, eval_dataloader
        cleanup_ddp()
    except OSError as e:
        if "[Errno 16] Device or resource busy" in str(e):
            print("DDP cleanup: Device or resource busy error, ignoring.")
        else:
            raise e


if __name__ == "__main__":
    main()
