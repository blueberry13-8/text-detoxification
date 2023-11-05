import torch


def decode_outputs(train_dataset, outputs, pad):
    sents = []
    for sent in outputs.detach().cpu():
        sent = torch.argmax(sent, dim=1)
        sent = train_dataset.vocab.lookup_tokens(sent.numpy())
        filtered_data = [item for item in sent if item != pad]
        sents.append(filtered_data)
    return sents


def decode_labels(train_dataset, labels, pad):
    sents = []
    for sent in labels.detach().cpu():
        sent = train_dataset.vocab.lookup_tokens(sent.numpy())
        filtered_data = [item for item in sent if item != pad]
        sents.append(filtered_data)
    return sents