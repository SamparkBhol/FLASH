import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import hashlib
import io

import src.config as config

class VulnerabilityModel(nn.Module):
    def __init__(self, input_size=config.NUM_FEATURES, output_size=1):
        super(VulnerabilityModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def get_model_weights(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}

def set_model_weights(model, weights):
    model.load_state_dict(weights)

def get_model_checksum(model_weights):
    m = hashlib.sha256()
    with io.BytesIO() as f:
        torch.save(model_weights, f)
        m.update(f.getvalue())
    return m.hexdigest()

def aggregate_weights(weighted_updates, total_weight):
    if not weighted_updates or total_weight == 0:
        return None

    global_weights = OrderedDict()
    first_update = weighted_updates[0][0]

    for key in first_update.keys():
        global_weights[key] = torch.zeros_like(first_update[key])

    for weights, weight in weighted_updates:
        for key in weights.keys():
            global_weights[key] += (weights[key] * weight) / total_weight

    return global_weights