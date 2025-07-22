from dataclasses import dataclass


@dataclass
class Hyperparameters:
    sequence_length: int = 10
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100

    def __str__(self):
        return f"Hyperparameters(sequence_length={self.sequence_length}, lstm_hidden_size={self.lstm_hidden_size}, lstm_num_layers={self.lstm_num_layers}, learning_rate={self.learning_rate}, batch_size={self.batch_size}, epochs={self.epochs})"