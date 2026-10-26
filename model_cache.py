"""Project-local pretrained models: download once via a mirror, then load offline."""

import hashlib
import json
import os
from pathlib import Path

# Set before transformers/diffusers import huggingface_hub.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
MODEL_ROOT = Path(__file__).resolve().parent / "model"


def _complete(directory, marker, source):
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        files = record["files"]
        return record["source"] == source and bool(files) and all(
            (directory / name).is_file()
            and (directory / name).stat().st_size == size
            for name, size in files.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def resolve_model(model_id, revision=None):
    """Return a local model directory; explicit paths never fall back to Hub."""
    model_id = str(model_id)
    local = Path(model_id).expanduser()
    if local.is_dir():
        return str(local.resolve())
    if local.exists() or local.is_absolute() or model_id.startswith((".", "~")):
        raise FileNotFoundError(f"Model directory does not exist: {local}")

    # Hash the full repo/revision pair so different checkpoints cannot collide.
    source = {"repo_id": model_id, "revision": revision or "main"}
    key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
    directory = MODEL_ROOT / key
    marker = directory / ".tiny-t2i-ready.json"
    if _complete(directory, marker, source):
        print(f"Using local model: {model_id} -> {directory}", flush=True)
        return str(directory)

    from filelock import FileLock
    from huggingface_hub import snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    # Serialize downloads of the same model, including simultaneous train/eval.
    with FileLock(str(MODEL_ROOT / f"{key}.lock")):
        if not _complete(directory, marker, source):
            marker.unlink(missing_ok=True)
            print(f"Downloading {model_id} via {os.environ['HF_ENDPOINT']} "
                  f"-> {directory}", flush=True)
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(directory),
                endpoint=os.environ["HF_ENDPOINT"],
                allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
            )
            # Write the completion marker only after a successful download.
            # Interrupted downloads are retried using Hub's local-dir metadata.
            files = {
                path.relative_to(directory).as_posix(): path.stat().st_size
                for path in directory.rglob("*")
                if path.is_file() and ".cache" not in path.relative_to(directory).parts
                and path.name != marker.name
            }
            if "config.json" not in files or not any(
                name.endswith((".safetensors", ".bin")) for name in files
            ):
                raise OSError(f"Downloaded model lacks config/weights: {directory}")
            marker.write_text(
                json.dumps({"source": source, "files": files}, indent=2),
                encoding="utf-8",
            )
    return str(directory)
