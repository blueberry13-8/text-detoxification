import requests
import os


def download_tsv():
    """
    Downloads a TSV file from a Yandex Disk public share and saves it locally.

    Example:
    download_tsv()

    Returns:
    None
    """
    # Skip downloading if file exists
    if os.path.exists('../../data/interim/filtered.tsv'):
        return
    public_key = 'https://disk.yandex.ru/d/HvdbdtPNl77LxQ'
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key='

    # Getting link for downloading from Yandex Disk
    final_url = base_url + public_key
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Load and save file
    download_response = requests.get(download_url)
    with open('../../data/interim/filtered.tsv', 'wb') as f:
        f.write(download_response.content)