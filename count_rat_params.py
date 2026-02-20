
import sys
import os
sys.path.append(os.getcwd())

from classifiers.differential.RAT import RAT, RAT_baseline, RAT_selfattn, RAT_zeroref
from extractors.XLSR import XLSR_300M
from feature_processors.MeanProcessor import MeanProcessor

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

print('Initializing models...')
# Instantiate the extractor and processor
# Manually freeze for accurate 'trainable' count
extractor = XLSR_300M(finetune=False)
for p in extractor.parameters():
    p.requires_grad = False

processor = MeanProcessor()

models_dict = {
    'RAT': RAT(extractor, processor, in_dim=1024),
    'RAT_baseline': RAT_baseline(extractor, processor, in_dim=1024),
    'RAT_selfattn': RAT_selfattn(extractor, processor, in_dim=1024),
    'RAT_zeroref': RAT_zeroref(extractor, processor, in_dim=1024)
}

print(f"{'Model':<15} | {'Trainable Params':<20} | {'Total Params':<20}")
print('-' * 60)

for name, model in models_dict.items():
    trainable, total = count_parameters(model)
    print(f"{name:<15} | {trainable:<20,} | {total:<20,}")
