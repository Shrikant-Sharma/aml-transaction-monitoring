"""Step 1. Get the six data files and check each one against its fingerprint.

    python 01_get_data.py            download what is missing (about 650 MB), then check everything
    python 01_get_data.py --check    only check the files already in data/raw/

The data is the IBM anti-money-laundering benchmark (Altman and others, NeurIPS 2023), released under the
Community Data License Agreement, Sharing, version 1.0. It is synthetic: no real customer or bank is in it.
The publisher's own page is on Kaggle and needs a sign-in:
    https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
This script downloads the same files from public copies on Hugging Face, which need no sign-in, and then
checks every file against the SHA-256 fingerprint recorded in data/manifest.json. If you would rather take the
files from Kaggle, put the six files named in the manifest into data/raw/ and run this script with --check.
"""
import hashlib
import sys
import urllib.request
import zipfile

from common import RAW, ROOT, read_json, sha256_of

MANIFEST = ROOT / "data" / "manifest.json"


def download(url, path):
    print(f"  downloading {path.name} ...", flush=True)
    part = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, shown = 0, 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total and done * 10 // total > shown:
                shown = done * 10 // total
                print(f"    {shown * 10}%", flush=True)
    part.replace(path)


def main():
    check_only = "--check" in sys.argv
    manifest = read_json(MANIFEST)
    RAW.mkdir(parents=True, exist_ok=True)
    if not check_only:
        for entry in manifest["files"]:
            path = RAW / entry["name"]
            if path.exists():
                continue
            if "inside" in entry:                                   # this file comes out of an archive
                archive = RAW / entry["inside"]["archive"]
                if not archive.exists():
                    download(entry["inside"]["url"], archive)
                print(f"  unpacking {entry['name']} ...", flush=True)
                with zipfile.ZipFile(archive) as z, z.open(entry["name"]) as src, open(path, "wb") as dst:
                    while True:
                        block = src.read(1 << 22)
                        if not block:
                            break
                        dst.write(block)
            else:
                download(entry["url"], path)
    bad = 0
    for entry in manifest["files"]:
        path = RAW / entry["name"]
        if not path.exists():
            print(f"MISSING   {entry['name']}")
            bad += 1
            continue
        same, note = sha256_of(path) == entry["sha256"], ""
        if not same and path.suffix == ".txt":                      # the same text saved with Windows line endings
            same = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == entry["sha256"]
            note = "  (Windows line endings, same text)" if same else ""
        bad += not same
        print(f"{'ok       ' if same else 'DIFFERENT'} {entry['name']}  ({path.stat().st_size:,} bytes){note}")
    if bad:
        print(f"\n{bad} file(s) missing or different from the manifest. The results in this folder were made with the files the manifest describes.")
        sys.exit(1)
    print("\nAll six files match the manifest.")


if __name__ == "__main__":
    main()
