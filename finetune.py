#!/usr/bin/env python3
import torch
from common import build_model, get_dataloaders, setup_ddp, cleanup_ddp, set_seed
from config import local_config
from parse_arguments import parse_args
from trainers.BaseFFTrainer import BaseFFTrainer


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
        if rank == 0:
            print(f"Loaded model from {args.checkpoint}.")
    else:
        raise ValueError("Checkpoint must be specified when only evaluating.")

    # Load the datasets
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
        print(f"Fine-tuning {type(model).__name__} on {type(train_dataloader.dataset).__name__} dataloader.")

    # Fine-tune the model
    if isinstance(trainer, BaseFFTrainer):
        trainer.finetune(train_dataloader, val_dataloader, eval_dataloader, numepochs=20, finetune_ssl=True)
        # trainer.eval(eval_dataloader, subtitle="finetune")
    else:
        raise NotImplementedError("Fine-tuning is only implemented for FF models.")
    
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
