from collections import Counter

class VocabTask2:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.size = 0

    def build_vocab(self, token_lists: list[list[str]], special_tokens: list[str], min_freq: int = 1):
        counter = Counter(w for seq in token_lists for w in seq)
        
        words = special_tokens.copy()
        for w, freq in counter.items():
            if freq >= min_freq and w not in special_tokens:
                words.append(w)
                
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for i, w in enumerate(words)}
        self.size = len(words)

    def encode(self, tokens: list[str]) -> list[int]:
        unk_idx = self.word2idx.get('<UNK>', 0)
        return [self.word2idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx2word.get(i, '') for i in indices]
