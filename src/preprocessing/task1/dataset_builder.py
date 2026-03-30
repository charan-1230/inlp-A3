from src.preprocessing.task1.padder import pad_sequences


def build_dataset(cipher_token_lists, plain_token_lists, cipher_vocab, plain_vocab):
    # Enforce maximum length
    MAX_LEN = 200

    cipher_encoded = [cipher_vocab.encode(seq)[:MAX_LEN] for seq in cipher_token_lists]
    plain_encoded = [plain_vocab.encode(seq)[:MAX_LEN] for seq in plain_token_lists]

    # No <SOS> or <EOS> tokens are added in sequence labeling!
    return cipher_encoded, plain_encoded


def pad_dataset(cipher_encoded, plain_encoded, cipher_vocab, plain_vocab):
    cipher_pad = cipher_vocab.char2idx['<PAD>']
    plain_pad = plain_vocab.char2idx['<PAD>']

    # Both encoded lists are token-aligned and have identical lengths per pair
    max_len = max((len(seq) for seq in cipher_encoded), default=200)

    cipher_padded, lengths = pad_sequences(
        cipher_encoded, cipher_pad, max_len
    )

    plain_padded, _ = pad_sequences(
        plain_encoded, plain_pad, max_len
    )

    return {
        "input": cipher_padded,
        "target": plain_padded,
        "lengths": lengths,
        "max_len": max_len
    }