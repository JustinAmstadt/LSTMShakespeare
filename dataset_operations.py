import torch

from hyperparameters import Hyperparameters


path = "tiny_shakespeare.txt"

def read_dataset():
    with open(path, 'r') as file:
        text = file.read()
    return text


def create_token_to_index_mapping():
    text = read_dataset()
    chars = sorted(list(set(text)))
    stoi = {s:i for i,s in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos, chars


stoi, itos, chars = create_token_to_index_mapping()


def create_char_sequences(text, seq_length=10, stride=1):
    X = []
    Y = []

    for i in range(0, len(text) - seq_length, stride):
        input_seq = text[i:i + seq_length]
        target_char = text[i + seq_length]
        X.append(encode(input_seq, stoi))
        Y.append(encode(target_char, stoi))

    return torch.tensor(X), torch.tensor(Y)


def split_dataset_chronologically(X, Y, train_ratio=0.8, dev_ratio=0.1):
    """
    Split dataset chronologically to maintain temporal order
    
    Args:
        X, Y: Input sequences and targets
        train_ratio: Proportion for training (default 0.8 = 80%)
        dev_ratio: Proportion for validation (default 0.1 = 10%)
        Remaining goes to test set
    
    Returns:
        (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test)
    """
    n_samples = len(X)
    
    train_end = int(n_samples * train_ratio)
    dev_end = int(n_samples * (train_ratio + dev_ratio))
    
    # Chronological splits
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_dev, Y_dev = X[train_end:dev_end], Y[train_end:dev_end]
    X_test, Y_test = X[dev_end:], Y[dev_end:]
    
    print(f"Dataset splits:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  Dev:   {len(X_dev)} samples ({len(X_dev)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
    
    return (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test)


def encode(text, stoi):
    if len(text) == 1:
        return stoi[text]
    else:
        return [stoi[c] for c in text]


def decode(tokens, itos):
    return ''.join([itos[i] for i in tokens])


# Create full dataset
text = read_dataset()
X, Y = create_char_sequences(text, seq_length=Hyperparameters.sequence_length, stride=1)

# Split into train/dev/test
(X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test) = split_dataset_chronologically(X, Y)

# Keep the original X, Y for backward compatibility, but now they represent the full dataset
# Use X_train, Y_train for training, X_dev, Y_dev for validation, X_test, Y_test for testing