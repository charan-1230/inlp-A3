"""
Tokenizers for sequence-labeling cipher decryption (Task 1).

Cipher structure:
  - '9' represents a space character
  - Every 2 consecutive non-9 digits → one plain text character

Tokenization:
  cipher_line  →  list of 2-digit tokens + '<SPACE>' for each '9'
  plain_line   →  list of individual characters (space kept as ' ')

After tokenization: len(cipher_tokens) == len(plain_tokens)
"""

SPACE_TOKEN = "<SPACE>"

def tokenize_cipher_line(line: str) -> list[str] | None:
    """
    Convert a raw cipher string into a flat list of tokens.

    '9'  → '<SPACE>'
    Every pair of non-9 digits → a 2-char token e.g. '31', '18'

    Returns None if the line cannot be cleanly tokenized (e.g. odd-length
    segment that would leave a dangling digit).
    """
    tokens = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '9':
            tokens.append(SPACE_TOKEN)
            i += 1
        else:
            # Expect exactly 2 digits
            if i + 1 >= len(line):
                return None          # dangling single digit — skip line
            pair = line[i:i + 2]
            if '9' in pair:
                return None          # '9' inside a 2-digit window — invalid
            tokens.append(pair)
            i += 2
    return tokens


def tokenize_plain_line(line: str) -> list[str]:
    """
    Split plain text into individual characters.
    Spaces are kept as the literal ' ' character.
    """
    return list(line)


def load_data(plain_path, cipher_path):
    with open(plain_path, 'r') as f:
        plain = [line.strip() for line in f.readlines()]

    with open(cipher_path, 'r') as f:
        cipher = [line.strip() for line in f.readlines()]

    assert len(plain) == len(cipher), "Mismatch in dataset!"

    return plain, cipher