def tokenize_line(line: str) -> list[str]:
    """
    Lowercase and tokenize a sentence by whitespace.
    Task 2 requirement: all text must be lowercased.
    """
    words = line.strip().lower().split(' ')
    return [w for w in words if w]
