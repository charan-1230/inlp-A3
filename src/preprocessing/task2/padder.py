def pad_sequence(seq: list[int], max_len: int, pad_idx: int) -> list[int]:
    """
    Pad sequence to exactly max_len. 
    If shorter, pad it.
    If longer, wait, chunking happens before this function.
    """
    if len(seq) < max_len:
        seq = seq + [pad_idx] * (max_len - len(seq))
    return seq
