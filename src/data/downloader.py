import requests
import os
import argparse


def download_tsv():
    """
    Downloads a TSV file from a Yandex Disk public share and saves it locally.

    Example:
    download_tsv()

    Returns:
    None
    """
    path = './data/interim/filtered.tsv'
    # Skip downloading if file exists
    if os.path.isfile(path):
        print('File already exists')
        return
    public_key = 'https://disk.yandex.ru/d/HvdbdtPNl77LxQ'
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key='

    # Getting link for downloading from Yandex Disk
    final_url = base_url + public_key
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Load and save file
    download_response = requests.get(download_url)
    with open(path, 'wb') as f:
        f.write(download_response.content)


def download_lstm_weights():
    """
    Downloads a .pt file of weights for lstm from a Yandex Disk public share and saves it locally.

    Example:
    download_lstm_weights()

    Returns:
    None
    """
    # Skip downloading if file exists
    path = './models/best_lstm.pt'
    if os.path.isfile(path):
        print('File already exists')
        return
    public_key = 'https://disk.yandex.ru/d/n3HUqi5oo1x44A'
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key='

    # Getting link for downloading from Yandex Disk
    final_url = base_url + public_key
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Load and save file
    download_response = requests.get(download_url)
    with open(path, 'wb') as f:
        f.write(download_response.content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("object", type=str)
    args = parser.parse_args()
    if args.object == 'paranmt_dataset':
        download_tsv()
    elif args.object == 'lstm_weights':
        download_lstm_weights()
    else:
        print('Invalid arguments')
