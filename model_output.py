import torch
import torch.nn.functional as F
from hyperparameters import Hyperparameters
from dataset_operations import chars, stoi, itos, encode, decode
from nn import CharLSTM


def make_model(model_string: str):
    hyperparams = Hyperparameters()
    model = CharLSTM(
        vocab_size=len(chars),
        hidden_size=hyperparams.lstm_hidden_size,
        num_layers=hyperparams.lstm_num_layers,
    )

    model.load_state_dict(torch.load(model_string))
    return model


def generate_text(model, seed_text, length=100, temperature=1.0):
    """
    Generate text using the trained LSTM model.
    
    Args:
        model: Trained CharLSTM model
        seed_text: Initial text to start generation (must be at least seq_length)
        length: Number of characters to generate
        temperature: Controls randomness (0.1=conservative, 1.0=balanced, 2.0=creative)
    
    Returns:
        Generated text string
    """
    model.eval()
    seq_length = Hyperparameters.sequence_length
    
    # Ensure seed text is long enough
    if len(seed_text) < seq_length:
        seed_text = seed_text.ljust(seq_length)
    
    # Take last seq_length characters if seed is too long
    current_seq = seed_text[-seq_length:]
    generated = seed_text
    
    with torch.no_grad():
        hidden = None
        
        for _ in range(length):
            # Encode current sequence
            input_indices = encode(current_seq, stoi)
            input_tensor = torch.tensor([input_indices])
            
            # Get model prediction
            output, hidden = model(input_tensor, hidden)
            
            # Get logits for the last character
            logits = output[0, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            char_idx = torch.multinomial(probs, 1).item()
            
            # Decode and append the character
            next_char = itos[int(char_idx)]
            generated += next_char
            
            # Update sequence for next iteration
            current_seq = current_seq[1:] + next_char
    
    return generated


if __name__ == "__main__":
    # Load the trained model
    model_path = "shakespeare_lstm_model-Hyperparameters-sequence_length=10-lstm_hidden_size=64-lstm_num_layers=1-learning_rate=0.0001-batch_size=32-epochs=100.pth"
    model = make_model(model_path)
    print("Model loaded successfully!\n")
    
    # Example 1: Generate text with different temperatures
    print("=== Text Generation Examples ===\n")
    
    seed_text = "ROMEO: How"
    print(f"Seed text: '{seed_text}'\n")
    
    # Try different temperature values
    temperatures = [0.5, 1.0, 1.5]
    for temp in temperatures:
        generated = generate_text(model, seed_text, length=200, temperature=temp)
        print(f"Temperature {temp}:")
        print(generated)
        print("-" * 80)
    
    # Example 2: Try different seed texts
    print("\n=== Different Seed Texts ===\n")
    
    seeds = [
        "First Citi",
        "To be or n",
        "JULIET:\n",
        "What is th"
    ]
    
    for seed in seeds:
        generated = generate_text(model, seed, length=400, temperature=1.0)
        print(f"Seed: '{seed}'")
        print(f"Generated: {generated}")
        print("-" * 80)