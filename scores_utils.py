#!/usr/bin/env python
from sys import argv
import os

import pandas as pd
from confidence_intervals import evaluate_with_conf_int

from trainers.utils import calculate_EER, calculate_minDCF

def calculate_metrics(path: str):
    df = pd.read_csv(path, sep=",", names=["file", "score", "label"])

    eer, eer_ci = evaluate_with_conf_int(samples = df["score"], metric = calculate_EER, labels = df["label"], num_bootstraps=1000)
    print(f"Metrics for {os.path.basename(path)}: EER: {eer*100:.2f}%, CI: [{eer_ci[0]*100:.2f}%, {eer_ci[1]*100:.2f}%]")

    mindcf, mindcf_ci = evaluate_with_conf_int(samples = df["score"], metric = calculate_minDCF, labels = df["label"], num_bootstraps=1000)
    print(f"Metrics for {os.path.basename(path)}: minDCF: {mindcf:.3f}, CI: [{mindcf_ci[0]:.3f}, {mindcf_ci[1]:.3f}]")


if __name__ == "__main__":
    calculate_metrics(argv[1])
