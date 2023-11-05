import argparse
from train_transformer import train_transformer
from train_lstm import train_lstm


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_num", type=int, default=10)
    args = parser.parse_args()

    if args.model_name == 'lstm':
        train_lstm(args.batch_size, args.epochs_num)
    elif args.model_name == 'transformer':
        train_transformer(args.batch_size, args.epochs_num)
    # elif args.model_name == 't5':
    #     print('Project is not supposed for T5 training')
