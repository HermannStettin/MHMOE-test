import os
import torch
from collections import Counter, OrderedDict

class Vocab(object):
    def __init__(
        self,
        special=[],
        min_freq=0,
        max_size=None,
        lower_case=True,
        delimiter=None,
        vocab_file=None,
    ):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def tokenize(
        self,
        line,
        add_eos=False,
    ):
        line = line.strip()
        if self.lower_case:
            line = line.lower()

        if self.delimiter == "":
            symbols = list(line)
        else:
            symbols = line.split(self.delimiter)

        if add_eos:
            return symbols + ["<eos>"]
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose: print(f"Counting file {path} ...")
        assert os.path.exists(path)
        sents = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)
        return sents

    def count_sents(self, sents, verbose=False):
        if verbose: print(f"Counting {len(sents)} sentences ...")
        for idx, symbols in enumerate(sents):
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        print(f"Building vocab from file: {vocab_file}")
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        if "<UNK>" not in self.sym2idx:
             print("Warning: <UNK> token not found in vocab file. Adding it.")
             self.add_special("<UNK>")
        self.unk_idx = self.sym2idx["<UNK>"]

    def build_vocab(self):
        if self.vocab_file:
            self._build_from_file(self.vocab_file)
        else:
            print(f"Building vocab with min_freq={self.min_freq}, max_size={self.max_size}")
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            if "<UNK>" not in self.sym2idx and "<UNK>" not in self.special:
                self.add_special("<UNK>")

        if hasattr(self, "unk_idx"):
            print(f"Final vocab size {len(self)} (UNK index: {self.unk_idx})")
        else:
            print(f"Final vocab size {len(self)}")


    def encode_file(self, path, ordered=False, verbose=False, add_eos=True):
        if verbose: print(f"Encoding file {path} ...")
        assert os.path.exists(path)
        encoded = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                symbols = self.tokenize(line, add_eos=add_eos)
                encoded.append(self.convert_to_tensor(symbols))
        if ordered:
            encoded = torch.cat(encoded)
        return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print(f"Encoding {len(sents)} sentences ...")
        encoded = [self.convert_to_tensor(symbols) for symbols in sents]
        if ordered:
            encoded = torch.cat(encoded)
        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            attr_name = "{}_idx".format(sym.strip("<>").lower().replace("-", "_"))
            setattr(self, attr_name, self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        if not 0 <= idx < len(self):
             print(f"Warning: Index {idx} out of vocab range [0, {len(self)-1}]. Returning <UNK>.")
             return self.idx2sym[self.unk_idx] if hasattr(self, "unk_idx") else "<???>"
        return self.idx2sym[idx]

    def get_idx(self, sym):
        return self.sym2idx.get(sym, self.unk_idx if hasattr(self, "unk_idx") else -1)

    def get_symbols(self, indices):
        return [self.get_sym(idx.item() if torch.is_tensor(idx) else idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude_idx=None):
        if exclude_idx is None: exclude_idx = []
        return " ".join([self.get_sym(idx) for idx in indices if idx not in exclude_idx])

    def __len__(self):
        return len(self.idx2sym)