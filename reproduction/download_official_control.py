"""Download pinned official LIBERO weights into an ignored project cache.

Checks exact advertised download size and keeps 1GiB disk headroom. Does not
delete existing artifacts. A download is not an evaluation result.
"""
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "cache/official_libero_spatial"
MODEL = "openvla/openvla-7b-finetuned-libero-spatial"


def main():
    info = HfApi().model_info(MODEL, files_metadata=True)
    files = [s for s in info.siblings if s.rfilename.endswith((".json", ".model", ".safetensors"))]
    total = sum(s.size for s in files)
    remaining = sum(s.size for s in files if not (DEST / s.rfilename).is_file()
                    or (DEST / s.rfilename).stat().st_size != s.size)
    free = shutil.disk_usage(ROOT).free
    if free < remaining + 2**30:
        raise RuntimeError(f"Insufficient disk space: {free} free, {remaining} download + 1GiB reserve required")
    print(json.dumps({"model": MODEL, "revision": info.sha, "bytes": total,
                      "remaining_download_bytes": remaining, "free_bytes": free}), flush=True)
    snapshot_download(MODEL, revision=info.sha, local_dir=DEST,
                      allow_patterns=[s.rfilename for s in files], max_workers=1)
    for file in files:
        assert (DEST / file.rfilename).stat().st_size == file.size, file.rfilename
    manifest = {"model": MODEL, "revision": info.sha, "total_bytes": total,
                "files": [{"path": s.rfilename, "bytes": s.size} for s in files],
                "status": "downloaded_not_evaluated"}
    evidence = ROOT / "reproduction/results/official_checkpoint_manifest.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    with evidence.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Official control checkpoint ready at {DEST}", flush=True)


if __name__ == "__main__":
    main()
