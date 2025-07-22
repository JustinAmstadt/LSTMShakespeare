import torch
import torch.nn as nn

from dataset_operations import (X_dev, X_test, X_train, Y_dev, Y_test, Y_train,
                                chars)


def _calculate_loss(criterion, output, targets):
    last_output = output[:, -1, :]
    last_output = last_output.view(-1, len(chars))
    return criterion(last_output, targets)


def _backward_and_optimize(model, optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


def _convert_targets(targets):
    return targets.view(-1)


def train_model(model, train_dataloader, criterion, optimizer, epochs=100):
    print(f"Train dataset size: {len(X_train)} sequences")
    print(f"Dev dataset size: {len(X_dev)} sequences")
    print(f"Test dataset size: {len(X_test)} sequences")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Sample input shape: {X_train[0].shape}, target: {Y_train[0]}")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        count = 0
        for inputs, targets in train_dataloader:
            # Inputs will be (batch_size, sequence_length)
            targets = _convert_targets(targets)

            # Forward pass
            output, _ = model.forward(inputs)
            loss = _calculate_loss(criterion, output, targets)
            _backward_and_optimize(model, optimizer, loss)

            total_loss += loss.item()
            num_batches += 1
            count += 1
            if count % 100 == 0:
                count = 0
                print(f"Epoch {epoch}, Loss: {total_loss/num_batches:.4f}")

        print(f"Epoch {epoch+1}, Average Loss: {total_loss/num_batches:.4f}")


def evaluate_model(model, dataloader, criterion):
    total_loss = 0
    num_batches = 0
    for inputs, targets in dataloader:
        targets = _convert_targets(targets)
        output, _ = model.forward(inputs)
        loss = _calculate_loss(criterion, output, targets)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def save_model(model, path):
    torch.save(model.state_dict(), path)
