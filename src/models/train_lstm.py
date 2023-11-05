import torch
from torch import nn
from dataset.make_loader import create_dataset_dataloader
from dataset.dataset import special_symbols, PAD_IDX
from lstm_model import Encoder, Decoder, Seq2Seq
import nltk
import matplotlib.pyplot as plt
from nltk.translate.meteor_score import single_meteor_score
from tqdm import tqdm
from torchtext.data.metrics import bleu_score


def train_lstm(batch_size=32, epochs=10):
    train_losses = []
    val_losses = []
    val_bleus = []
    val_meteors = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Data preparing
    print('---Data preparation: in progress...')
    train_dataset, val_dataset, train_dataloader, val_dataloader = create_dataset_dataloader(batch_size)
    print('---Data preparation: Done!')

    print('---Model creation: in progress...')
    VOCAB_DIM = len(train_dataset.vocab)
    EMB_DIM = 96
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    enc = Encoder(VOCAB_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    dec = Decoder(VOCAB_DIM, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print('--Model creation: Done!')

    nltk.download('wordnet')
    print('--Training and Validation')

    num_epochs = epochs
    for epoch in range(num_epochs):
        train_one_epoch(model, train_dataloader, optimizer, criterion, train_losses, epoch)
        torch.save(model.state_dict(), f'./src/models/checkpoints/lstm_{epoch}.pt')
        val_one_epoch(model, val_dataloader, criterion, val_losses, val_bleus, val_meteors, train_dataset, epoch)

    plot_losses(train_losses, val_losses, val_bleus, val_meteors)
    print('Done. After each epoch model weights were saved to the src/models/checkpoints/')


def train_one_epoch(model,
                    loader,
                    optimizer,
                    loss_fn,
                    train_losses,
                    epoch_num=-1):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",
        leave=True,
    )
    model.train()
    train_loss = 0.0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, batch in loop:
        texts, labels = batch
        texts = texts.to(device)
        labels = labels.to(device)

        outputs = model(texts, labels).to(device)
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        labels = labels[:, 1:].reshape(-1)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += len(batch)
        loop.set_postfix({"loss": train_loss / total, "batch loss": loss.item() / len(batch)})
    train_losses.append(train_loss / total)


def val_one_epoch(
        model,
        loader,
        loss_fn,
        val_losses,
        val_bleus,
        val_meteors,
        train_dataset,
        epoch_num=-1
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: validation",
        leave=True,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0.0
    total = 0
    reference_corpus = []  # List to store reference sentences
    candidate_corpus = []  # List to store candidate sentences
    for i, batch in loop:
        texts, labels = batch
        texts = texts.to(device)
        labels = labels.to(device)

        outputs = model(texts, labels).to(device)

        # Convert the model's output to sentences
        predicted_sentences = decode_outputs(train_dataset, outputs)
        reference_sentences = decode_labels(train_dataset, labels)

        # Append reference and candidate sentences for BLEU score computation
        reference_corpus.extend(reference_sentences)
        candidate_corpus.extend(predicted_sentences)

        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        labels = labels[:, 1:].reshape(-1)

        loss = loss_fn(outputs, labels)

        test_loss += loss.item()
        total += len(batch)
        loop.set_postfix({"total loss": test_loss / total, "batch loss": loss.item() / len(batch)})
    val_losses.append(test_loss / total)
    valid_bleu = bleu_score(candidate_corpus, reference_corpus)
    mean_meteor = 0
    for i in range(len(reference_corpus)):
        mean_meteor += single_meteor_score(reference_corpus[i], candidate_corpus[i])
    mean_meteor /= len(candidate_corpus)
    val_bleus.append(valid_bleu)
    val_meteors.append(mean_meteor)
    print('BLEU:', valid_bleu, 'METEOR:', mean_meteor)


def plot_losses(train_losses, val_losses, val_bleus, val_meteors):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot(range(len(train_losses)), train_losses, label='training', marker='o', linestyle='-')
    ax1.plot(range(len(val_losses)), val_losses, label='validation', marker='o', linestyle='-')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoches')
    ax1.legend()

    ax2.plot(range(len(val_bleus)), val_bleus, marker='o', linestyle='-')
    ax2.set_ylabel('BLEU')
    ax2.set_xlabel('Epoches')

    ax3.plot(range(len(val_meteors)), val_meteors, marker='o', linestyle='-')
    ax3.set_ylabel('METEOR')
    ax3.set_xlabel('Epoches')

    plt.show()


def decode_outputs(train_dataset, outputs):
    sents = []
    for sent in outputs.detach().cpu():
        sent = torch.argmax(sent, dim=1)
        sent = train_dataset.vocab.lookup_tokens(sent.numpy())
        filtered_data = [item for item in sent if item != special_symbols[PAD_IDX]]
        sents.append(filtered_data)
    return sents


def decode_labels(train_dataset, labels):
    sents = []
    for sent in labels.detach().cpu():
        sent = train_dataset.vocab.lookup_tokens(sent.numpy())
        filtered_data = [item for item in sent if item != special_symbols[PAD_IDX]]
        sents.append(filtered_data)
    return sents
