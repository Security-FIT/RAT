#!/usr/bin/env python3
from argparse import Namespace
import torch

from common import build_model

from torchaudio import load


def load_model_for_interactive():
    args: Namespace = Namespace()
    args.extractor = "XLSR_300M"
    args.classifier = "RAT"
    args.processor = "Mean"
    model, trainer = build_model(args)
    
    return model.eval()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_wf, sr1 = load("fake.flac")
    r_wf, sr2 = load("fake.flac")
    # wf, sr = load("babis-zeman.mp3")

    model = load_model_for_interactive()
    print(model(torch.vstack([r_wf, r_wf]).to(device), torch.vstack([f_wf, f_wf]).to(device)))
