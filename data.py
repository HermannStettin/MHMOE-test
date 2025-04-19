import os
import torch

def _tokenize(text_path, dictionary_to_update):
    """Tokenizes a text file."""
    print("Tokenizing {}".format(text_path))
    assert os.path.exists(text_path)

    nb_tokens_in_dictionary = len(dictionary_to_update)

    with open(text_path, "r", encoding="utf8") as f:
        for line in f:
            # Assumes space tokenization and adds <eos>
            tokens = line.split() + ["<eos>"]
            for token in tokens:
                if token not in dictionary_to_update:
                    dictionary_to_update[token] = nb_tokens_in_dictionary
                    nb_tokens_in_dictionary += 1

    ids = []
    with open(text_path, "r", encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ["<eos>"]
            for token in tokens:
                ids.append(dictionary_to_update[token])
    ids = torch.LongTensor(ids)
    return ids


class Corpus:
    """Builds a dictionary and tokenizes train/valid/test files."""
    def __init__(self, data_path):
        self._dictionary = {}
        self.train = _tokenize(
            text_path=os.path.join(data_path, "train.txt"),
            dictionary_to_update=self._dictionary,
        )
        self.valid = _tokenize(
            text_path=os.path.join(data_path, "valid.txt"),
            dictionary_to_update=self._dictionary,
        )
        self.test = _tokenize(
            text_path=os.path.join(data_path, "test.txt"),
            dictionary_to_update=self._dictionary,
        )

    @property
    def vocab_size(self):
        return len(self._dictionary)


def _batchify(data_tensor, batch_size):
    """Reshapes data into [batch_size, sequence_length]."""
    nb_batches = data_tensor.size(0) // batch_size
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def _build_corpus(data_path, env_params):
    """Builds or loads the corpus, handles distributed building."""
    corpus_path = os.path.join(data_path, "corpus.pt")
    rank = env_params.get("rank", 0)
    distributed = env_params.get("distributed", False)

    if os.path.exists(corpus_path):
        if not distributed or rank == 0: print(f"Loading existing corpus: {corpus_path}")
        corpus = torch.load(corpus_path, map_location='cpu')
    else:
        if not distributed or rank == 0: print(f"Building corpus: {corpus_path}")
        if not distributed or rank == 0:
            corpus = Corpus(data_path)
            torch.save(corpus, corpus_path)
            if distributed:
                 torch.distributed.barrier()
        else:
             torch.distributed.barrier()
             corpus = torch.load(corpus_path, map_location='cpu')

    return corpus


def _get_train_val_test_data(corpus, batch_size):
    """Batchifies the corpus data."""
    return [
        _batchify(corpus.train, batch_size),
        _batchify(corpus.valid, batch_size),
        _batchify(corpus.test, batch_size),
    ]


def get_train_val_test_data(data_params, env_params, batch_size, device):
    """Main function to get processed train, validation, and test data."""
    corpus = _build_corpus(data_params["data_path"], env_params=env_params)
    data_params["vocab_size"] = corpus.vocab_size

    train_data, val_data, test_data = _get_train_val_test_data(
        corpus=corpus, batch_size=batch_size
    )

    if env_params.get("distributed", False):
        world_size = env_params["world_size"]
        rank = env_params["rank"]
        global_bs = batch_size

        if global_bs % world_size != 0:
             raise ValueError(f"Global batch size {global_bs} must be divisible by world_size {world_size}")
        device_batch_size = global_bs // world_size

        slice_data = slice(rank * device_batch_size, (rank + 1) * device_batch_size)
        train_data = train_data[slice_data]
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    return train_data, val_data, test_data