import pickle
import torch as pt


def decode(indices, vocab):
    chars = []
    for idx in indices:
        if idx in vocab.idx2char:
            token = vocab.idx2char[idx]
            if token not in ["<PAD>", "<UNK>"]:
                # print literally so we can see spaces
                chars.append(token)
    return "".join(chars)


def main():
    # Load vocabs
    with open("data/processed/task1/cipher_vocab.pkl", "rb") as f:
        cipher_vocab = pickle.load(f)

    with open("data/processed/task1/plain_vocab.pkl", "rb") as f:
        plain_vocab = pickle.load(f)

    # Print vocab
    print("Cipher vocab:")
    for token, idx in cipher_vocab.char2idx.items():
        print(f"{token}: {idx}")

    print("\nPlain vocab:")
    for token, idx in plain_vocab.char2idx.items():
        print(f"{token}: {idx}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = pt.load(
        "data/processed/task1/dataset.pt",
        map_location="cpu",
        weights_only=False
    )

    cipher = dataset["input"]
    plain = dataset["target"]

    print("\nDataset Shapes:")
    print("Cipher:", cipher.shape)
    print("Plain:", plain.shape)

    print("\nSample dataset lines + decoding check:\n")

    n = min(10, len(cipher))

    for i in range(n):
        cipher_line = cipher[i]
        plain_line = plain[i]

        print(f"--- Sample {i} ---")

        print("Cipher indices:", cipher_line.tolist())
        print("Plain indices :", plain_line.tolist())

        # Decode
        decoded_cipher = decode(cipher_line.tolist(), cipher_vocab)
        decoded_plain = decode(plain_line.tolist(), plain_vocab)

        print("Decoded Cipher:", decoded_cipher[:100])
        print("Decoded Plain :", decoded_plain[:100])

        # 🔍 Sanity checks

        # 1. Padding check
        pad_idx = plain_vocab.char2idx["<PAD>"]
        if pad_idx in plain_line:
            first_pad = (plain_line == pad_idx).nonzero(as_tuple=True)[0][0].item()
            assert all(x == pad_idx for x in plain_line[first_pad:]), " Bad padding!"

        # 2. Space <SPACE> token check (new alignment check)
        space_count = decoded_plain.count(" ")
        token_space_count = decoded_cipher.count("<SPACE>")

        print(f"Plain Spaces: {space_count}, Cipher <SPACE>s: {token_space_count}")

        print("-" * 40)


if __name__ == "__main__":
    main()