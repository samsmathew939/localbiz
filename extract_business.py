"""Extract only yelp_academic_dataset_business.json from the downloaded archive.

Usage (from localbiz/, venv python, run isolated):
    python -I extract_business.py data/Yelp-JSON.zip data

Streams the nested tar straight out of the zip and writes the single business
file to <outdir>/yelp_academic_dataset_business.json using a sanitized basename
(no path traversal). Treats the archive as untrusted data.
"""
import sys, os, zipfile, tarfile

TARGET = "yelp_academic_dataset_business.json"


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python -I extract_business.py <zip> <outdir>")
    zip_path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, TARGET)  # fixed safe name, not from archive

    with zipfile.ZipFile(zip_path) as z:
        tar_name = next((n for n in z.namelist() if n.endswith(".tar")), None)
        if not tar_name:
            sys.exit("no .tar found inside zip")
        tar_path = os.path.join(outdir, "yelp_dataset.tar")
        print(f"Extracting {tar_name} to disk ...")
        with z.open(tar_name) as src, open(tar_path, "wb") as f:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  wrote {tar_path} ({os.path.getsize(tar_path):,} bytes)")

    found = False
    with tarfile.open(tar_path, mode="r") as tf:
        for m in tf:
            if m.isreg() and os.path.basename(m.name) == TARGET:
                print(f"Found {m.name} ({m.size:,} bytes) — extracting ...")
                src = tf.extractfile(m)
                with open(out, "wb") as f:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"Wrote {out} ({os.path.getsize(out):,} bytes)")
                found = True
                break
    os.remove(tar_path)  # after the tar is closed (Windows locks open files)
    print(f"Cleaned up {tar_path}")
    if not found:
        sys.exit(f"{TARGET} not found in archive")


if __name__ == "__main__":
    main()
