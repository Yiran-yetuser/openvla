"""Fetch exactly one public LIBERO task file with a pinned revision/LFS hash.

No model or training data is uploaded. Existing complete files are verified and
reused; partial files are preserved and never silently restarted or overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
FILENAME = "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5"
SIZE = 632345992
DIGEST = "9d0b5435b313a8c0d7336b429b5f0f578be6f2a9f5f8a6dd67459ef388f3c74c"
DESTINATION = ROOT / "cache/libero_expert" / FILENAME
MANIFEST = ROOT / "reproduction/results/expert_demo_manifest_v1.json"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILENAME}"


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"public_source": URL, "bytes": SIZE, "sha256": DIGEST,
                          "destination": str(DESTINATION)}, indent=2))
        return
    if DESTINATION.exists():
        if DESTINATION.stat().st_size != SIZE or digest_file(DESTINATION) != DIGEST:
            raise ValueError("Existing expert file fails hash/size; preserve and diagnose")
    else:
        partial = DESTINATION.with_suffix(".hdf5.part")
        if partial.exists():
            raise FileExistsError("Prior partial preserved; no automatic repeated download")
        if shutil.disk_usage(ROOT).free < SIZE + 3 * 2**30:
            raise RuntimeError("Insufficient disk reserve; no deletion or download")
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        # Public unauthenticated request; do not forward Hub tokens to redirects.
        with requests.get(URL, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with partial.open("xb") as handle:
                for chunk in response.iter_content(2**20):
                    count += len(chunk)
                    if count > SIZE:
                        raise ValueError("Download exceeds pinned size")
                    handle.write(chunk)
                    digest.update(chunk)
                    if count // (64 * 2**20) != (count - len(chunk)) // (64 * 2**20):
                        print(f"expert file: {count}/{SIZE} bytes", flush=True)
        if count != SIZE or digest.hexdigest() != DIGEST:
            raise ValueError("Download hash/size mismatch; partial preserved")
        partial.rename(DESTINATION)
    result = {"schema_version": 1, "status": "verified_pinned_public_expert_file",
              "repo_id": REPO, "revision": REVISION, "file": FILENAME, "url": URL,
              "local_file": str(DESTINATION.relative_to(ROOT)), "bytes": SIZE, "sha256": DIGEST,
              "source_reference": "https://github.com/Lifelong-Robot-Learning/LIBERO#datasets",
              "limits": ["Original LIBERO HDF5, not the OpenVLA-regenerated no-noop RLDS file.",
                         "This manifest does not establish TFDS episode correspondence or successful replay."]}
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != result:
            raise ValueError("Existing manifest mismatch")
    else:
        with MANIFEST.open("x") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps({"status": result["status"], "bytes": SIZE, "sha256": DIGEST}, indent=2))


if __name__ == "__main__":
    main()
