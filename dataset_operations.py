path = "tiny_shakespeare.txt"

def read_dataset():
    with open(path, 'r') as file:
        text = file.read()
    return text


def create_token_to_index_mapping():
    text = read_dataset()
    chars = sorted(list(set(text)))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos


stoi, itos = create_token_to_index_mapping()


def create_char_sequences(text, seq_length=10, stride=1):
    X = []
    Y = []

    for i in range(0, len(text) - seq_length, stride):
        input_seq = text[i:i + seq_length]
        target_char = text[i + seq_length]
        X.append(encode(input_seq, stoi))
        Y.append(encode(target_char, stoi))
        # print(input_seq, target_char)

    return X, Y


def encode(text, stoi):
    if len(text) == 1:
        return stoi[text]
    else:
        return [stoi[c] for c in text]


def decode(tokens, itos):
    return ''.join([itos[i] for i in tokens])


text = read_dataset()
X, Y = create_char_sequences(text, seq_length=10, stride=1)
print(X[:10])
print(Y[:10])