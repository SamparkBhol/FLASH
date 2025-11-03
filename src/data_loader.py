import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

import src.config as config

def generate_synthetic_data(n_samples, n_features):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=2,
        random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def create_client_loaders(num_clients, n_samples_per_client, n_features, test_split=0.2):
    total_samples = num_clients * n_samples_per_client
    X, y = generate_synthetic_data(total_samples, n_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )

    client_data_indices = np.random.dirichlet(
        [config.DATA_SPLIT_ALPHA] * num_clients,
        size=len(y_train)
    ).argmax(axis=1)

    client_loaders = []
    for i in range(num_clients):
        client_indices = np.where(client_data_indices == i)[0]
        if len(client_indices) == 0:
            client_loaders.append(None)
            continue

        X_client = torch.tensor(X_train[client_indices], dtype=torch.float32)
        y_client = torch.tensor(y_train[client_indices], dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_client, y_client)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return client_loaders, test_loader