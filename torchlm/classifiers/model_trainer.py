import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import LabelEncoder
from torchlm.models import *
from tqdm import tqdm
import torchmetrics

class Classifier:
    def __init__(self, model, device, use_mixed_precision = False, log_dir = None):
    