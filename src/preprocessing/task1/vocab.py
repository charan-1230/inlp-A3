class Vocab:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def __contains__(self, item):
        return item in self.char2idx
    
    def __getitem__(self, char):
        return self.char2idx[char]
    
    def get(self, char, default=None):
        return self.char2idx.get(char, default)
    
    def items(self):
        return self.char2idx.items()
    
    def build_vocab(self, token_lists: list[list[str]], special_tokens: list[str] = None):
        """
        Build vocabulary from tokenized sequences.
        `token_lists` is a list of lists of strings (tokens).
        `special_tokens` forces <PAD>, <SPACE>, etc. at the front.
        """
        chars = set()
        for tokens in token_lists:
            chars.update(tokens)

        chars = sorted(list(chars))

        if special_tokens:
            # Ensure we don't duplicate special tokens if they already exist in data
            chars = [c for c in chars if c not in special_tokens]
            chars = special_tokens + chars

        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.size = len(chars)

    def encode(self, tokens: list[str]) -> list[int]:
        # Assign undefined tokens to <UNK>, or <PAD> as a last resort
        unk = self.char2idx.get('<UNK>', self.char2idx.get('<PAD>', 0))
        return [self.char2idx.get(t, unk) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx2char.get(i, "") for i in indices]