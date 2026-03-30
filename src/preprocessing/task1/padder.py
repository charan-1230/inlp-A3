def add_special_tokens(seq, sos_idx, eos_idx):
    return [sos_idx] + seq + [eos_idx]


def pad_sequences(sequences, pad_idx, max_len):
    padded = []
    lengths = []

    for seq in sequences:
        lengths.append(min(len(seq), max_len))
        if len(seq) < max_len:
            seq = seq + [pad_idx] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)

    return padded, lengths