import argparse


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('model_name', type=str)
    parser.add_argument('weights_path', type=str)
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()

    if args.model_name == 'lstm':
        pass
    if args.model_name == 'transformer':
        pass
    if args.model_name == 't5':
        pass