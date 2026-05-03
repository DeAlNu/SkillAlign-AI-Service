import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.nlp_preprocessor import NLPPreprocessor
from src.preprocessing.embeddings import EmbeddingManager
from src.models.model_architecture import SkillAlignMatcher
from src.models.custom_loss import focal_loss
from src.training.train import ModelTrainer
from src.utils.metrics import compute_all_metrics, compute_classification_report, check_performance_targets
from src.utils.visualization import TrainingVisualizer

print(f'TF: {tf.__version__}')
print('Imports OK!')
