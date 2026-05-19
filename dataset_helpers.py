import os
import tarfile
import requests
from pathlib import Path

def download_and_extract(url, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = dest_dir / filename
    if not filepath.exists():
        print(f"Downloading {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    print(f"Extracting {filepath}...")
    if str(filepath).endswith(".tar.gz") or str(filepath).endswith(".tgz"):
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=dest_dir)
    elif str(filepath).endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    print("Done.")

def download_common_voice(dest_dir):
    url = "https://voice.mozilla.org/en/datasets"  # Replace with direct download link for your language/version
    # Example: url = "https://datasets-server.huggingface.co/1.0/parquet/mozilla-foundation/common_voice_11_0/en/0.parquet"
    # You may need to manually fetch the latest .tar.gz link for your language
    print("Please manually download Common Voice from:", url)

def download_librispeech(dest_dir, subset="test-clean"):
    urls = {
        "test-clean": "http://www.openslr.org/resources/12/test-clean.tar.gz",
        "train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "train-clean-360": "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "dev-clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    }
    url = urls[subset]
    download_and_extract(url, dest_dir)