# All the global path informations for the proejct

import os
import sys

# Get the current file's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.abspath('/mnt/qb/work/oh/owl336/thesis/CLIP-not-BoW-unimodally/')
# Get dataset directory
dataset_dir = os.path.join(current_dir, 'datasets')
SPEC_DATASET_DIR = os.path.join(dataset_dir, 'SPEC')
WHATSUP_DATASET_DIR = os.path.join(dataset_dir, 'WhatsUp')