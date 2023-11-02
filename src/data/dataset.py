import torch
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from collections import Counter
import torchtext
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd

# Define special indices for vocabulary
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class DeToxicityDataset(Dataset):
    def __init__(self, dataframe, to_remove_word_cnt=2, vocab=None, tox_diff=0.9):
        """
        Initialize a DeToxicityDataset.

        Args:
            dataframe (pandas.DataFrame): The input DataFrame containing toxic text data.
            to_remove_word_cnt (int): The count of occurrences to remove less frequent words from the vocabulary.
            vocab (torchtext.legacy.data.Field): A predefined vocabulary for tokenization (optional).
            tox_diff (float): The toxicity difference threshold for filtering data.
        """
        self.df = dataframe
        self._preprocess_sentences(to_remove_word_cnt, tox_diff)
        assert len(self.references) == len(self.translations)
        self.vocab = vocab or self._create_vocab()

    def _preprocess_sentences(self, to_remove_word_cnt, tox_diff):
        """
        Preprocess the sentences in the input DataFrame.

        Args:
            to_remove_word_cnt (int): The count of occurrences to remove less frequent words from the vocabulary.
            tox_diff (float): The toxicity difference threshold for filtering data.
        """
        # Swap all ref with trn where toxicity level is greater in ref
        to_swap = self.df['ref_tox'] < self.df['trn_tox']
        self.df.loc[to_swap, ['reference', 'translation']] = self.df.loc[to_swap, ['translation', 'reference']].values
        self.df.loc[to_swap, ['ref_tox', 'trn_tox']] = self.df.loc[to_swap, ['trn_tox', 'ref_tox']].values

        # Delete all rows where difference between ref_tox and trn_tox is less than tox_diff
        self.df = self.df[self.df['ref_tox'] - self.df['trn_tox'] >= tox_diff]

        self.df.loc[:, 'reference'] = self.df['reference'].apply(lambda text: text.lower())
        self.df.loc[:, 'translation'] = self.df['translation'].apply(lambda text: text.lower())

        # Tokenize sentences
        self.df['tokenized_reference'] = self.df['reference'].apply(lambda text: word_tokenize(text))
        self.df['tokenized_translation'] = self.df['translation'].apply(lambda text: word_tokenize(text))

        # Collect all words and count their occurrence in sentences
        all_sent = self.df['tokenized_translation'].tolist() + self.df['tokenized_reference'].tolist()
        all_words = [word for sent in all_sent for word in sent]
        token_counts = Counter(all_words)

        # Remove all words which occur less or equal than 'to_remove_word_cnt'
        self.unique_words = set(all_words)
        for word in token_counts:
            if token_counts[word] <= to_remove_word_cnt:
                self.unique_words.remove(word)

        # Leave only approved words in tokenized sentences
        self.df['tokenized_reference'] = self.df['tokenized_reference'].apply(
            lambda tokens: [word for word in tokens if word in self.unique_words])
        self.df['tokenized_translation'] = self.df['tokenized_translation'].apply(
            lambda tokens: [word for word in tokens if word in self.unique_words])

        self.references = self.df['tokenized_reference'].tolist()
        self.translations = self.df['tokenized_translation'].tolist()

    def _create_vocab(self):
        """
        Create a vocabulary based on unique words in the preprocessed data.

        Returns:
            torchtext.legacy.data.Field: A vocabulary object.
        """
        vocab = torchtext.vocab.vocab(Counter(list(self.unique_words)), specials=special_symbols)
        vocab.set_default_index(0)

        return vocab

    def _get_reference(self, index: int) -> list:
        """
        Retrieve a reference sentence from the dataset by index and tokenize it.

        Args:
            index (int): The index of the reference sentence.

        Returns:
            list: A list of tokenized reference sentence.
        """
        sent = self.references[index]
        return self.vocab(sent)

    def _get_translation(self, index: int) -> list:

        """
        Retrieve a translation sentence from the dataset by index and tokenize it.

        Args:
            index (int): The index of the translation sentence.

        Returns:
            list: A list of tokenized translation sentence.
        """
        sent = self.translations[index]
        return self.vocab(sent)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index) -> tuple[list, list]:
        return self._get_reference(index), self._get_translation(index)


def collate_batch(batch: list):
    """
    Collate a batch of data, pad sequences, and prepare tensors for training.

    Args:
        batch (list): A list of data samples, where each sample is a tuple of reference and translation sentences.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Padded reference and translation tensors.
    """
    max_size = 50
    references_batch, translations_batch = [], []
    for _ref, _trn in batch:
        _ref, _trn = [BOS_IDX] + _ref[:max_size - 2] + [EOS_IDX], [BOS_IDX] + _trn[:max_size - 2] + [EOS_IDX]
        if len(_ref) < max_size:
            _ref = [PAD_IDX] * (max_size - len(_ref)) + _ref
        if len(_trn) < max_size:
            _trn = [PAD_IDX] * (max_size - len(_trn)) + _trn
        references_batch.append(torch.tensor(_ref))
        translations_batch.append(torch.tensor(_trn))

    return torch.stack(references_batch), torch.stack(translations_batch)


def create_dataset_dataloader():
    """
    Create a dataset and corresponding data loaders for training and validation.

    Returns:
        tuple[DeToxicityDataset, DeToxicityDataset, DataLoader, DataLoader]: Train dataset, validation dataset,
        train data loader, and validation data loader.
    """
    extracted_dir = '../../data/interim/'
    tsv_path = extracted_dir + 'filtered.tsv'
    tsv_file = pd.read_csv(tsv_path, sep='\t', index_col=0)

    validation_ratio = 0.2
    train_dataframe, val_dataframe = train_test_split(tsv_file, test_size=validation_ratio, random_state=123)

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    train_dataset = DeToxicityDataset(train_dataframe)
    val_dataset = DeToxicityDataset(val_dataframe, vocab=train_dataset.vocab)

    batch_size = 32

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_dataset, val_dataset, train_dataloader, val_dataloader
