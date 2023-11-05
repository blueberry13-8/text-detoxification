from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from collections import Counter
import torchtext


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
