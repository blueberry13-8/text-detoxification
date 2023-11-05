from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
import torch
import nltk
from dataset import PAD_IDX, BOS_IDX, EOS_IDX, DeToxicityDataset


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


def create_dataset_dataloader(batch_size=32):
    """
    Create a dataset and corresponding data loaders for training and validation.

    Returns:
        tuple[DeToxicityDataset, DeToxicityDataset, DataLoader, DataLoader]: Train dataset, validation dataset,
        train data loader, and validation data loader.
    """
    extracted_dir = './data/interim/'
    tsv_path = extracted_dir + 'filtered.tsv'
    tsv_file = pd.read_csv(tsv_path, sep='\t', index_col=0)

    validation_ratio = 0.2
    train_dataframe, val_dataframe = train_test_split(tsv_file, test_size=validation_ratio, random_state=123)

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    train_dataset = DeToxicityDataset(train_dataframe)
    val_dataset = DeToxicityDataset(val_dataframe, vocab=train_dataset.vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_dataset, val_dataset, train_dataloader, val_dataloader
