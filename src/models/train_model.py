import argparse
from train_lstm import train_lstm


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()

    if args.model_name == 'lstm':
        train_lstm()
    elif args.model_name == 'transformer':
        pass
    # elif args.model_name == 't5':
    #     print('Project is not supposed for T5 training')
