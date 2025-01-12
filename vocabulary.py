from collections import Counter
import nltk
nltk.download('punkt')


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.word2idx)


def build_vocab(captions_file, threshold=5):
    counter = Counter()
    with open(captions_file, "r") as f:
        for line in f:
            _, _, caption = line.strip().split(",")
            words = caption.lower().split()
            counter.update(words)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    for word in words:
        vocab.add_word(word)

    return vocab
