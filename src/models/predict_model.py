import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset.dataset import BOS_IDX, EOS_IDX, PAD_IDX
import nltk
from archs.lstm_model import Encoder, Decoder, Seq2Seq
from archs.custom_transformer import Transformer


def translate_sentence_lstm(model, sentence, vocab, max_length=50):
    """
    Translate a sentence using the LSTM-based model.

    Args:
        model (Seq2Seq): The LSTM-based sequence-to-sequence model.
        sentence (list): The input sentence as a list of tokens.
        vocab (torchtext.legacy.data.Field): The vocabulary used for tokenization.
        max_length (int): The maximum length for the generated translation.

    Returns:
        str: The translated sentence.
    """
    sentence_tensor = torch.tensor(sentence).unsqueeze(0)
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)
    outputs = [BOS_IDX]
    for i in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]])
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)
        if best_guess == EOS_IDX:
            break
    translated_sentence = vocab.lookup_tokens(outputs)
    # remove start token
    return ' '.join(translated_sentence[1:-1])


def translate_sentence_transformer(model, sentence, vocab, max_length=50):
    """
    Translate a sentence using the Transformer-based model.

    Args:
        model (Transformer): The Transformer-based model.
        sentence (list): The input sentence as a list of tokens.
        vocab (torchtext.legacy.data.Field): The vocabulary used for tokenization.
        max_length (int): The maximum length for the generated translation.

    Returns:
        str: The translated sentence.
    """
    sentence_tensor = torch.tensor(sentence).unsqueeze(0)
    outputs = [BOS_IDX]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)
        if best_guess == EOS_IDX:
            break
    translated_sentence = vocab.lookup_tokens(outputs)
    # remove start token
    return ' '.join(translated_sentence[1:-1])


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('model_name', type=str)
    parser.add_argument('input_file_path', type=str)
    args = parser.parse_args()

    # Read input text from the file
    input_text = open(args.input_file_path, 'r').read().split('\n')
    output_file = open('translation.txt', 'w')
    if args.model_name == 't5':
        # Use the T5 model for translation
        tokenizer = AutoTokenizer.from_pretrained("s-nlp/t5-paranmt-detox")
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path='./models')
        for i in range(len(input_text)):
            input_ids = tokenizer(
                input_text[i], return_tensors="pt"
            ).input_ids
            output = model.generate(input_ids, max_new_tokens=50)
            tran = tokenizer.decode(output[0], skip_special_tokens=True)
            output_file.write(tran + '\n')
        output_file.close()
        exit()

    vocab = None
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    model = None
    if args.model_name == 'lstm':
        # Use the LSTM-based model
        vocab = torch.load('src/models/utils/vocab_lstm.pth')
        VOCAB_DIM = 37097
        EMB_DIM = 96
        HID_DIM = 512
        N_LAYERS = 2
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        enc = Encoder(VOCAB_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(VOCAB_DIM, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
        model = Seq2Seq(enc, dec, 'cpu')
        ckpt = torch.load("./models/best_lstm.pt", 'cpu')
        model.load_state_dict(ckpt)
    if args.model_name == 'transformer':
        # Use the Transformer-based model
        vocab = torch.load('src/models/utils/vocab_transformer.pth')
        INPUT_DIM = 37155
        EMB_DIM = 96
        N_HEAD = 4
        N_LAYERS = 3
        DROPOUT = 0.10
        model = Transformer(EMB_DIM, INPUT_DIM, PAD_IDX, N_HEAD, N_LAYERS, N_LAYERS, DROPOUT)
        ckpt = torch.load("./models/best_transformer.pt", 'cpu')
        model.load_state_dict(ckpt)

    input_text = [vocab(nltk.word_tokenize(text)) for text in input_text]

    for i in range(len(input_text)):
        translation = ''
        if args.model_name == 'lstm':
            translation = translate_sentence_lstm(model, input_text[i], vocab)
        elif args.model_name == 'transformer':
            translation = translate_sentence_transformer(model, input_text[i], vocab)
        output_file.write(translation + '\n')
    output_file.close()
