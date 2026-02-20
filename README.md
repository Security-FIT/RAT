# Reference-Augmented Training for ASV Anti-Spoofing

**Authors:** [ANONYMIZED]

**Abstract:** We introduce a spoofing countermeasure architecture conditioned on speaker-reference recordings, but observe that it converges to a solution that effectively ignores the reference during inference. Surprisingly, training with a reference channel induces invariance that improves deepfake detection, even when the reference is absent or mismatched during inference. Based on this observation, we propose a Reference-Augmented Training (RAT) strategy. RAT yields improved detection performance compared to single-utterance baselines, even when the reference recording is replaced with a zero vector at inference. Through rigorous analysis, we demonstrate that the optimization process rapidly diminishes the reference contributions, leading to a graceful disconnection of the reference channel. Using RAT, we achieve state-of-the-art 2.57\% EER and 0.074 minDCF on the ASVspoof 5 benchmark, surpassing even large ensembles with just a single detector.

### Reproducibility 
This repository contains the code for our paper Reference-Augmented Training for ASV Anti-Spoofing (link available after publishing). The model used in the paper is available at [Google Drive for download](https://drive.google.com/file/d/1C85iclcmch9Mvxcg_myGiy5EAgo73Dyl/view?usp=sharing). Scores are available in the `scores` folder.

## Repository structure

```
RAT
├── augmentation        <- contains various data augmentation techniques
├── classifiers         <- contains the classes for models
│   ├── differential        <- pair-input
│   └── single_input        <- single-input
├── datasets            <- contains Dataset classes (ASVspoof (2019, 2021), ASVspoof5, In-the-Wild, Morphing)
├── extractors          <- contains various feature extractors
├── feature_processors  <- contains pooling implementation (MeanProcessor)
├── scores              <- contains scored ASVspoof 5 evaluation in csv format [file_name, score, label]
├── trainers            <- contains classes for training and evaluating models
├ README.md
├ common.py             <- common code, enums, maps, dataloaders
├ config.py             <- hardcoded config, paths, batch size
├ eval.py               <- script for evaluating trained model under reference degradations
├ eval.py               <- script for evaluating trained model
├ finetune.py           <- script for finetuning trained model including the SSL backend
├ load_model_for_interactive.py <- script for building the model and loading the weights
├ parse_arguments.py    <- argument parsing script
├ requirements.txt      <- requirements to install in conda environment
├ scores_utils.py       <- functions for score analysis and evaluation
└ train_and_eval.py     <- main script for training and evaluating models
```

## Requirements

**Python 3.10**, possibly works with newer versions\
**PyTorch >2.2.0** including torchvision and torchaudio \
packages in `requirements.txt`

Simply install the required conda environment with:

```
# optional, create and activate conda env
# conda create -n diff_detection python=3.10
# conda activate diff_detection

# install required packages
# !!always refer to pytorch website https://pytorch.org/ for up-to-date command!!
# conda install pytorch torchvision torchaudio cpuonly -c pytorch  # For CPU-only install
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia  # For GPU-enabled install

pip install -r requirements.txt
```

## Usage

Based on the use-case, use either `train_and_eval.py` or `eval.py` scripts with the following arguments:

```
usage: 
train_and_eval.py [-h/--help] --local [--checkpoint CHECKPOINT] -d DATASET -e EXTRACTOR -p PROCESSOR -c CLASSIFIER [-a/--augment] [-ep NUM_EPOCHS] [--seed SEED]

Main script for training and evaluating the classifiers.

options:
  -h, --help            show this help message and exit

  --local               Flag for running locally.

  --checkpoint CHECKPOINT
                        Path to a checkpoint to be loaded. If not specified, the model will be trained from scratch.

  -d DATASET, --dataset DATASET
                        Dataset to be used. See common.DATASETS for available datasets.

  -e EXTRACTOR, --extractor EXTRACTOR
                        Extractor to be used. See common.EXTRACTORS for available extractors.

  -p PROCESSOR, --processor PROCESSOR, --pooling PROCESSOR
                        Feature processor to be used. One of: Mean

  -c CLASSIFIER, --classifier CLASSIFIER
                        Classifier to be used. See common.CLASSIFIERS for available classifiers.

Optional arguments:
  -a, --augment         Flag for using data augmentation defined in augmentation/Augment.py

Classifier-specific arguments:
  -ep NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of epochs to train for.
```

## Multi-GPU Training & Evaluation

The scripts support multi-GPU training and evaluation Distributed Data Parallel (DDP) via `torchrun`. To run the training on multiple GPUs (e.g., 2 GPUs), use the following command:

```bash
torchrun --standalone --nproc_per_node=2 train_and_eval.py ...
```

Ensure that you have `torchrun` installed (it comes with `torch`) and that the available GPUs are visible to the process (e.g. `CUDA_VISIBLE_DEVICES=0,1`).

## Contact

For any inquiries, questions or ask for help/explanation, contact me at [ANONYMIZED].
