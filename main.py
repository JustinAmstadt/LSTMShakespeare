import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_operations import (X_dev, X_test, X_train, Y_dev, Y_test, Y_train,
                                chars)
from device import get_device
from hyperparameters import Hyperparameters
from nn import CharLSTM
from train import evaluate_model, save_model, train_model


def _create_dataloaders(X, Y):
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=Hyperparameters.batch_size, shuffle=False)


def _make_model():
    device = get_device()
    model = CharLSTM(
        vocab_size=len(chars),
        hidden_size=Hyperparameters.lstm_hidden_size,
        num_layers=Hyperparameters.lstm_num_layers,
    )
    model.to(device)
    return model


def make_dataloaders():
    train_dataloader = _create_dataloaders(X_train, Y_train)
    dev_dataloader = _create_dataloaders(X_dev, Y_dev)
    test_dataloader = _create_dataloaders(X_test, Y_test)
    return train_dataloader, dev_dataloader, test_dataloader


def main():
    model = _make_model()
    train_dataloader, dev_dataloader, test_dataloader = make_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Hyperparameters.learning_rate)

    train_model(model, train_dataloader, criterion, optimizer)
    test_loss = evaluate_model(model, test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}")

    hyperparameters = Hyperparameters()

    save_model(model, path=f"shakespeare_lstm_model-{str(hyperparameters)}.pth")


if __name__ == "__main__":
    main()
