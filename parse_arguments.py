import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Main script for training and evaluating the classifiers.")

    group = parser.add_mutually_exclusive_group(required=True)
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
    feature_processors = ["Mean"]
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

    # region Optional arguments
    # training
    parser.add_argument(
        "-ep",
        "--num_epochs",
        type=int,
        help="Number of epochs to train for.",
        default=20,
    )

    # seed
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for reproducibility.",
        default=42,
    )

    # endregion

    args = parser.parse_args()

    return args
