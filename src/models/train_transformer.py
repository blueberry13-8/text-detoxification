import torch
from torch import nn
from dataset.make_loader import create_dataset_dataloader
from dataset.dataset import special_symbols, PAD_IDX
from archs.custom_transformer import Transformer
import nltk
from utils.plots import plot_losses
from utils.decoders import decode_outputs, decode_labels
from nltk.translate.meteor_score import single_meteor_score
from tqdm import tqdm
from torchtext.data.metrics import bleu_score


def train_transformer(batch_size=32, epochs=10):
    torch.manual_seed(123)
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
    INPUT_DIM = len(train_dataset.vocab)
    EMB_DIM = 96
    N_HEAD = 4
    N_LAYERS = 3
    DROPOUT = 0.10
    model = Transformer(EMB_DIM, INPUT_DIM, PAD_IDX, N_HEAD, N_LAYERS, N_LAYERS, DROPOUT).to(device)
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
        inp_data = batch[0].to(device)
        target = batch[1].to(device)

        output = model(inp_data, target[:, :-1])

        output = output.reshape(-1, output.shape[2])
        target = target[:, 1:].reshape(-1)

        loss = loss_fn(output, target)

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
        inp_data = batch[0].to(device)
        target = batch[1].to(device)

        output = model(inp_data, target[:, :-1])

        # Convert the model's output to sentences
        predicted_sentences = decode_outputs(output, train_dataset, special_symbols[PAD_IDX])
        reference_sentences = decode_labels(target, train_dataset, special_symbols[PAD_IDX])

        # Append reference and candidate sentences for BLEU score computation
        reference_corpus.extend(reference_sentences)
        candidate_corpus.extend(predicted_sentences)

        output = output.reshape(-1, output.shape[2])
        target = target[:, 1:].reshape(-1)

        loss = loss_fn(output, target)

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
