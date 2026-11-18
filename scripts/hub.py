"""Save results to a private Hugging Face repository, and get them back.

Colab's Google Drive mount lost large files that had already been written
(see docs/DECISIONS.md), so trained models and long-run results are stored in a
private Hugging Face dataset repository instead. Uploads either finish or fail
loudly, and every push is checked against the files the Hub actually stores.

Repository paths mirror project paths (e.g. outputs/adapters/agents_8k).

    export HF_TOKEN=...   # a token with write access
    python scripts/hub.py push --root /content/work outputs/adapters/agents_8k judges/large/sanctity
    python scripts/hub.py pull --root /content/work outputs/adapters/agents_8k
    python scripts/hub.py list

The default repository is <your username>/moral-rag-cf-artifacts (private).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_TYPE = "dataset"


def local_files(root: Path, rel: str) -> dict[str, int]:
    """{repo path: size} for a file or every file under a folder."""
    target = root / rel
    if target.is_file():
        return {rel: target.stat().st_size}
    if target.is_dir():
        return {f"{rel}/{p.relative_to(target).as_posix()}": p.stat().st_size
                for p in sorted(target.rglob("*")) if p.is_file()}
    raise FileNotFoundError(f"Nothing to push at {target}")


def remote_files(api, repo: str, rel: str) -> dict[str, int]:
    """{repo path: size} of what the Hub stores under rel (or rel itself)."""
    from huggingface_hub.hf_api import RepoFile

    out = {}
    try:
        for item in api.list_repo_tree(repo, path_in_repo=rel, recursive=True, repo_type=REPO_TYPE):
            if isinstance(item, RepoFile):
                out[item.path] = item.size
    except Exception:
        pass
    if not out:  # rel may be a single file, which list_repo_tree does not list
        try:
            for info in api.get_paths_info(repo, [rel], repo_type=REPO_TYPE):
                if isinstance(info, RepoFile):
                    out[info.path] = info.size
        except Exception:
            pass
    return out


def check(expected: dict[str, int], stored: dict[str, int]) -> list[str]:
    """Paths that are missing on the Hub or have the wrong size."""
    return [p for p, size in expected.items() if stored.get(p) != size]


def push(api, repo: str, root: Path, rel: str) -> bool:
    expected = local_files(root, rel)
    target = root / rel
    if target.is_dir():
        api.upload_folder(folder_path=str(target), path_in_repo=rel, repo_id=repo,
                          repo_type=REPO_TYPE, commit_message=f"Save {rel}")
    else:
        api.upload_file(path_or_fileobj=str(target), path_in_repo=rel, repo_id=repo,
                        repo_type=REPO_TYPE, commit_message=f"Save {rel}")
    bad = check(expected, remote_files(api, repo, rel))
    total = sum(expected.values()) / 2**20
    if bad:
        print(f"PROBLEM  {rel}: {len(bad)} of {len(expected)} files missing or incomplete on the Hub, e.g. {bad[:3]}")
        return False
    print(f"SAVED    {rel}  ({len(expected)} files, {total:.0f} MB, checked on the Hub)")
    return True


def pull(repo: str, root: Path, rels: list[str]) -> None:
    from huggingface_hub import snapshot_download

    patterns = [p for rel in rels for p in (rel, f"{rel}/*")]
    snapshot_download(repo_id=repo, repo_type=REPO_TYPE, allow_patterns=patterns, local_dir=str(root))
    for rel in rels:
        status = "OK     " if (root / rel).exists() else "MISSING"
        print(f"{status}  {rel}")


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["push", "pull", "list"])
    parser.add_argument("paths", nargs="*", help="project-relative paths, e.g. outputs/adapters/agents_8k")
    parser.add_argument("--root", default=".", help="local folder the paths are relative to")
    parser.add_argument("--repo", default=os.environ.get("HF_REPO"), help="default: <user>/moral-rag-cf-artifacts")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set. Run the token cell first.")
    api = HfApi()
    repo = args.repo or f"{api.whoami()['name']}/moral-rag-cf-artifacts"
    root = Path(args.root)

    if args.action == "list":
        from huggingface_hub.hf_api import RepoFile

        for item in api.list_repo_tree(repo, recursive=True, repo_type=REPO_TYPE):
            if isinstance(item, RepoFile):
                print(f"{item.size / 2**20:>10.1f} MB  {item.path}")
        return
    if not args.paths:
        sys.exit("Give at least one path.")
    if args.action == "pull":
        pull(repo, root, args.paths)
        return

    api.create_repo(repo, repo_type=REPO_TYPE, private=True, exist_ok=True)
    ok = all([push(api, repo, root, rel) for rel in args.paths])
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
