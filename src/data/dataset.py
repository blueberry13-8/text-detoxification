import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import torch


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class DeToxicityDataset(Dataset):
    def __init__(self, dataframe, to_remove_word_cnt=5, vocab=None, tox_diff=0.3):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.df = dataframe
        self._preprocess_sentences(to_remove_word_cnt, tox_diff)
        assert len(self.references) == len(self.translations)
        self.vocab = vocab or self._create_vocab()

    def _preprocess_sentences(self, to_remove_word_cnt, tox_diff):
        # Swap all ref with trn where toxicity level is greater in ref
        to_swap = self.df['ref_tox'] < self.df['trn_tox']
        self.df.loc[to_swap, ['reference', 'translation']] = self.df.loc[to_swap, ['translation', 'reference']].values
        self.df.loc[to_swap, ['ref_tox', 'trn_tox']] = self.df.loc[to_swap, ['trn_tox', 'ref_tox']].values

        # Delete all rows where difference between ref_tox and trn_tox is less than tox_diff
        self.df = self.df[self.df['ref_tox'] - self.df['trn_tox'] >= tox_diff]

        # Tokenize sentences
        self.df['tokenized_reference'] = self.df['reference'].apply(lambda text: word_tokenize(text))
        self.df['tokenized_translation'] = self.df['translation'].apply(lambda text: word_tokenize(text))

        # Collect all words and count their occurrence in sentences
        all_sent = self.df['tokenized_translation'].tolist() + self.df['tokenized_reference'].tolist()
        all_words = [word for sent in all_sent for word in sent]
        token_counts = Counter(all_words)

        # Remove all words which occur less or equal than 'to_remove_word_cnt'
        unique_words = set(all_words)
        for word in token_counts:
            if token_counts[word] <= to_remove_word_cnt:
                unique_words.remove(word)

        # Leave only approved words in tokenized sentences
        self.df['tokenized_reference'] = self.df['tokenized_reference'].apply(
            lambda tokens: [word for word in tokens if word in unique_words])
        self.df['tokenized_translation'] = self.df['tokenized_translation'].apply(
            lambda tokens: [word for word in tokens if word in unique_words])

        # self.df = self.df[self.df['tokenized_reference'].apply(lambda x: len(x) <= max_sent_len)]
        # self.df = self.df[self.df['tokenized_translation'].apply(lambda x: len(x) <= max_sent_len)]
        # self.df['tokenized_reference'] = self.df['tokenized_reference'].apply(lambda tokens: [special_symbols[2]] + tokens + [special_symbols[3]])
        # self.df['tokenized_translation'] = self.df['tokenized_translation'].apply(lambda tokens: [special_symbols[2]] + tokens + [special_symbols[3]])
        self.references = self.df['tokenized_reference'].tolist()
        self.translations = self.df['tokenized_translation'].tolist()

    def _create_vocab(self):
        # creates vocabulary that is used for encoding
        # the sequence of tokens (splitted sentence)
        vocab = build_vocab_from_iterator(self.references + self.translations, specials=special_symbols)
        vocab.set_default_index(UNK_IDX)
        return vocab

    def _get_reference(self, index: int) -> list:
        # retrieves sentence from dataset by index
        sent = self.references[index]
        return self.vocab(sent)

    def _get_translation(self, index: int) -> list:
        # retrieves translation from dataset by index
        sent = self.translations[index]
        return self.vocab(sent)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index) -> tuple[list, list]:
        return self._get_reference(index), self._get_translation(index)


def create_dataset_dataloader():
    extracted_dir = '../'
    tsv_path = extracted_dir + 'filtered.tsv'
    tsv_file = pd.read_csv(tsv_path, sep='\t', index_col=0)

    validation_ratio = 0.2
    train_dataframe, val_dataframe = train_test_split(tsv_file, test_size=validation_ratio, random_state=123)

    train_dataset = DeToxicityDataset(train_dataframe)
    val_dataset = DeToxicityDataset(val_dataframe, vocab=train_dataset.vocab)

    batch_size = 128

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


max_size = 50


def collate_batch(batch: list):
    references_batch, translations_batch = [], []
    for _ref, _trn in batch:
        _ref, _trn = _ref[:max_size], _trn[:max_size]
        if len(_ref) < max_size:
            _ref = [PAD_IDX] * (max_size - len(_ref)) + _ref
        if len(_trn) < max_size:
            _trn = [PAD_IDX] * (max_size - len(_trn)) + _trn
        references_batch.append(torch.tensor(_ref))
        translations_batch.append(torch.tensor(_trn))

    return torch.stack(references_batch), torch.stack(translations_batch)



