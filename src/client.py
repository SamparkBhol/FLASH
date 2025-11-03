import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

import src.config as config
from src.model import VulnerabilityModel, get_model_weights, set_model_weights

class Client:
    def __init__(self, client_id, data_loader, is_malicious=False, sidechain_id=0):
        self.client_id = client_id
        self.data_loader = data_loader
        self.is_malicious = is_malicious
        self.sidechain_id = sidechain_id
        self.model = VulnerabilityModel().to(config.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.BCELoss()
        self.num_samples = len(data_loader.dataset) if data_loader else 0

    def train(self, global_weights):
        set_model_weights(self.model, global_weights)
        self.model.train()

        if not self.data_loader:
            return get_model_weights(self.model), OrderedDict(), 0

        for epoch in range(config.EPOCHS_PER_CLIENT):
            for features, labels in self.data_loader:
                features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)

                if self.is_malicious:
                    labels = 1 - labels

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        local_weights = get_model_weights(self.model)
        weight_update_delta = OrderedDict()

        for key in global_weights.keys():
            update = local_weights[key] - global_weights[key]

            if self.is_malicious:
                update = update * config.POISON_FACTOR

            noise = torch.randn_like(update) * config.DP_NOISE_STD_DEV
            weight_update_delta[key] = update + noise

        return local_weights, weight_update_delta, self.num_samples