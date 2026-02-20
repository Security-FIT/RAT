#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torchaudio
import os
from torch.utils.data import DataLoader

# Import existing modules
from config import local_config
from common import build_model, get_dataloaders, setup_ddp, cleanup_ddp, DATASETS, CLASSIFIERS, set_seed
from datasets.ASVspoof5 import ASVspoof5Dataset_pair

# Helper function for degradation
def apply_degradation(waveform, sample_rate, duration, snr, silence, noise_only):
    # 1. Restrict duration
    if duration is not None:
        target_samples = int(duration * sample_rate)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
    
    # 2. Pure Silence
    if silence:
        return torch.zeros_like(waveform)

    # 3. Noise Only
    if noise_only:
        # Generate noise with same power as original signal
        signal_power = torch.mean(waveform ** 2)
        noise = torch.randn_like(waveform)
        if signal_power > 0:
            current_noise_power = torch.mean(noise ** 2)
            if current_noise_power > 0:
                scale = torch.sqrt(signal_power / current_noise_power)
                return noise * scale
        return noise

    # 4. Add noise
    if snr is not None:
        # Calculate signal power
        signal_power = torch.mean(waveform ** 2)
        if signal_power > 0:
            snr_linear = 10 ** (snr / 10)
            noise_power = signal_power / snr_linear
            noise = torch.randn_like(waveform)
            current_noise_power = torch.mean(noise ** 2)
            if current_noise_power > 0:
                scale = torch.sqrt(noise_power / current_noise_power)
                waveform = waveform + noise * scale
    return waveform

# Define the new dataset classes
class ASVspoof5Dataset_pair_degraded(ASVspoof5Dataset_pair):
    GLOBAL_REF_DURATION = None
    GLOBAL_REF_SNR = None
    GLOBAL_REF_SILENCE = False
    GLOBAL_REF_NOISE_ONLY = False
    GLOBAL_REF_MISMATCH = False

    def __init__(self, root_dir, protocol_file_name, variant="train", augment=False, rir_root=""):
        super().__init__(root_dir, protocol_file_name, variant, augment, rir_root)
        self.ref_duration = self.GLOBAL_REF_DURATION
        self.ref_snr = self.GLOBAL_REF_SNR
        self.ref_silence = self.GLOBAL_REF_SILENCE
        self.ref_noise_only = self.GLOBAL_REF_NOISE_ONLY
        self.ref_mismatch = self.GLOBAL_REF_MISMATCH

        if self.ref_mismatch:
             bonafide_df = self.protocol_df[self.protocol_df["KEY"] == "bonafide"]
             self.bonafide_speakers = bonafide_df["SPEAKER_ID"].unique()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker_id = self.protocol_df.loc[idx, "SPEAKER_ID"]

        test_audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        test_audio_name = os.path.join(self.rec_dir, f"{test_audio_file_name}.flac")
        test_waveform, sample_rate = torchaudio.load(test_audio_name)

        label = self.protocol_df.loc[idx, "KEY"]
        label = 0 if label == "bonafide" else 1

        if self.ref_mismatch:
            # Select a different speaker who has bonafide recordings
            other_speakers = self.bonafide_speakers[self.bonafide_speakers != speaker_id]
            if len(other_speakers) > 0:
                mismatched_speaker_id = np.random.choice(other_speakers)
                speaker_recordings_df = self.protocol_df[
                    (self.protocol_df["SPEAKER_ID"] == mismatched_speaker_id) & (self.protocol_df["KEY"] == "bonafide")
                ]
            else:
                # Fallback to same speaker if no others available (should not happen in standard datasets)
                speaker_recordings_df = self.protocol_df[
                    (self.protocol_df["SPEAKER_ID"] == speaker_id) & (self.protocol_df["KEY"] == "bonafide")
                ]
        else:
            # Get the genuine speech of the same speaker for differentiation
            speaker_recordings_df = self.protocol_df[
                (self.protocol_df["SPEAKER_ID"] == speaker_id) & (self.protocol_df["KEY"] == "bonafide")
            ]

        if speaker_recordings_df.empty:
            raise Exception(f"Speaker {speaker_id} genuine speech not found in protocol file")

        # Get a random genuine speech of the speaker using sample()
        gt_audio_file_name = speaker_recordings_df.sample(n=1).iloc[0]["AUDIO_FILE_NAME"]
        gt_audio_name = os.path.join(self.rec_dir, f"{gt_audio_file_name}.flac")
        gt_waveform, _ = torchaudio.load(gt_audio_name)

        # Apply degradation
        gt_waveform = apply_degradation(gt_waveform, sample_rate, self.ref_duration, self.ref_snr, self.ref_silence, self.ref_noise_only)

        if self.augment:
            test_waveform = self.augmentor.augment(test_waveform)
            gt_waveform = self.augmentor.augment(gt_waveform)

        return test_audio_file_name, gt_waveform, test_waveform, label

def parse_args():
    parser = argparse.ArgumentParser(description="Main script for training and evaluating the classifiers (Degraded).")

    # either --metacentrum, --sge or --local must be specified
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metacentrum", action="store_true", help="Flag for running on metacentrum.")
    group.add_argument("--sge", action="store_true", help="Flag for running on SGE on BUT FIT.")
    group.add_argument("--local", action="store_true", help="Flag for running locally.")

    # Add argument for loading a checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint to be loaded. If not specified, the model will be trained from scratch.",
    )

    # dataset
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="ASVspoof2019LADataset_pair",
        help="Dataset to be used. See common.DATASETS for available datasets.",
        required=True,
    )

    # extractor
    parser.add_argument(
        "-e",
        "--extractor",
        type=str,
        default="XLSR_300M",
        help=f"Extractor to be used. See common.EXTRACTORS for available extractors.",
        required=True,
    )

    # feature processor
    feature_processors = ["MHFA", "AASIST", "Mean", "SLS"]
    parser.add_argument(
        "-p",
        "--processor",
        "--pooling",
        type=str,
        help=f"Feature processor to be used. One of: {', '.join(feature_processors)}",
        required=True,
    )

    # classifier
    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        help=f"Classifier to be used. See common.CLASSIFIERS for available classifiers.",
        required=True,
    )

    # augmentations
    parser.add_argument(
        "-a",
        "--augment",
        action="store_true",
        help="Flag for whether to use augmentations during training. Does nothing during evaluation.",
    )

    # Add arguments specific to each classifier
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    classifier_args = parser.add_argument_group("Classifier-specific arguments")
    for classifier, (classifier_class, args) in CLASSIFIERS.items():
        if args:  # if there are any arguments that can be passed to the classifier
            for arg, arg_type in args.items():
                if arg == "kernel":  # only for SVMDiff, display the possible kernels
                    classifier_args.add_argument(
                        f"--{arg}",
                        type=str,
                        help=f"{arg} for {classifier}. One of: {', '.join(kernels)}",
                    )
                else:
                    classifier_args.add_argument(f"--{arg}", type=arg_type, help=f"{arg} for {classifier}")

    classifier_args.add_argument(
        "-ep",
        "--num_epochs",
        type=int,
        help="Number of epochs to train for.",
        default=20,
    )
    
    # seed
    classifier_args.add_argument(
        "--seed",
        type=int,
        help="Seed for reproducibility.",
        default=42,
    )

    # New arguments
    parser.add_argument("--ref-duration", type=float, help="Duration limit for reference recordings in seconds (e.g., 1.0 or 3.0)")
    parser.add_argument("--ref-snr", type=float, help="SNR for additive noise on reference recordings in dB (e.g., 10.0 or 20.0)")
    parser.add_argument("--ref-silence", action="store_true", help="Replace reference recordings with silence")
    parser.add_argument("--ref-noise-only", action="store_true", help="Replace reference recordings with random noise (preserving energy)")
    parser.add_argument("--ref-mismatch", action="store_true", help="Use a reference recording from a different speaker")

    return parser.parse_args()

def main():
    args = parse_args()

    set_seed(args.seed)

    is_ddp, rank, local_rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    config = local_config

    # Set global variables for the datasets
    ASVspoof5Dataset_pair_degraded.GLOBAL_REF_DURATION = args.ref_duration
    ASVspoof5Dataset_pair_degraded.GLOBAL_REF_SNR = args.ref_snr
    ASVspoof5Dataset_pair_degraded.GLOBAL_REF_SILENCE = args.ref_silence
    ASVspoof5Dataset_pair_degraded.GLOBAL_REF_NOISE_ONLY = args.ref_noise_only
    ASVspoof5Dataset_pair_degraded.GLOBAL_REF_MISMATCH = args.ref_mismatch

    # Monkeypatch the dataset classes
    DATASETS["ASVspoof5Dataset_pair"] = ASVspoof5Dataset_pair_degraded

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
        distributed=is_ddp,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    assert isinstance(
        eval_dataloader, DataLoader
    ), "Error type of eval_dataloader returned from get_dataloaders."

    if rank == 0:
        print(
            f"Evaluating {args.checkpoint} {type(model).__name__} on "
            + f"{type(eval_dataloader.dataset).__name__} dataloader."
        )
        if args.ref_duration:
            print(f"Reference duration restricted to {args.ref_duration}s")
        if args.ref_snr:
            print(f"Reference SNR set to {args.ref_snr}dB")
        if args.ref_silence:
            print("Reference recordings replaced with silence")
        if args.ref_noise_only:
            print("Reference recordings replaced with random noise")
        if args.ref_mismatch:
            print("Reference recordings selected from mismatched speakers")

    trainer.eval(eval_dataloader, subtitle=str(args.checkpoint))
    
    del eval_dataloader
    cleanup_ddp()


if __name__ == "__main__":
    main()
