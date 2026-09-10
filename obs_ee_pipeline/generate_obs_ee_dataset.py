"""Regenerate the entire adityx23/icl-dataset locally, in the same LeRobot
v2.1 layout, with two new columns added to every episode's parquet:
`observation.left_ee` / `observation.right_ee` -- the FK counterpart of
generate_full_dataset.py's `action.q_target`, computed from the recorded
`observation.state` (real joint encoders) via ik_pipeline.compute_obs_ee
(see vr_teleop_ik.md §9).

Everything else is carried through unchanged: every existing parquet column,
meta/tasks.jsonl, meta/episodes.jsonl verbatim, and meta/episodes_stats.jsonl
with per-episode min/max/mean/std/count for the two new fields merged into
each episode's existing stats entry. meta/info.json is copied with just the
new features appended to "features". Videos are NOT duplicated here -- this
output has no videos/ directory; the source dataset's videos still apply
unchanged per episode_index (see README.md).

Sharding: same modulo scheme as generate_full_dataset.py (episode_index %
num_shards), for the same reason -- see that file's docstring. Each shard
writes its own parquet files (no overlap, safe in parallel) plus a
shard-local `meta/episodes_stats.partial.shard{N}-of-{M}.jsonl` (also no
overlap). Run `--merge-stats` once after every shard finishes.

Run inside the `lerobot` conda env: `conda activate lerobot`.

Usage:
    python generate_obs_ee_dataset.py                                   # all episodes, one process
    python generate_obs_ee_dataset.py --shard 0 --num-shards 6          # this shard's 1/6 slice
    python generate_obs_ee_dataset.py --merge-stats --num-shards 6      # combine after all shards finish
    python generate_obs_ee_dataset.py --start 0 --end 50                # contiguous slice, for testing
    python generate_obs_ee_dataset.py --out output/icl-dataset-obs-ee
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.ik_pipeline import EE_POSE_NAMES, HUB_REVISION, REPO_ID_DEFAULT, compute_obs_ee, get_metadata, load_model


def _download_meta_file(repo_id: str, rel_path: str) -> Path:
    return Path(hf_hub_download(repo_id, rel_path, repo_type="dataset", revision=HUB_REVISION))


def _prepare_meta(repo_id: str, out_dir: Path) -> dict:
    """Writes the meta files that carry over unchanged (idempotent -- safe to
    call from multiple concurrently-started shards since every shard writes
    identical bytes; only the per-shard stats need real merge handling, see
    merge_stats). Returns the parsed info.json dict (ee-pose features added,
    not yet written -- merge_stats writes the final info.json)."""
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    for name in ("meta/tasks.jsonl", "meta/episodes.jsonl"):
        dest = out_dir / name
        if not dest.exists():
            src = _download_meta_file(repo_id, name)
            dest.write_bytes(src.read_bytes())

    info_path = _download_meta_file(repo_id, "meta/info.json")
    info = json.loads(info_path.read_text())
    for key in ("observation.left_ee", "observation.right_ee"):
        info["features"][key] = {
            "dtype": "float32",
            "shape": [7],
            "names": EE_POSE_NAMES,
        }
    return info


def _stats_for(arr: np.ndarray) -> dict:
    """arr: (N, D) float array -> per-dim min/max/mean/std, count=[N] — same
    shape convention as the source dataset's episodes_stats.jsonl entries."""
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [arr.shape[0]],
    }


def generate(repo_id: str, out_dir: Path, episodes: list[int], shard_label: str) -> None:
    """Regenerates just this shard's `episodes` (any subset -- modulo-sharded,
    contiguous, or a single index). Writes each episode's parquet directly
    (no cross-shard overlap) and this shard's stats as
    meta/episodes_stats.partial.{shard_label}.jsonl -- combine with
    merge_stats once every shard is done."""
    meta = get_metadata(repo_id)
    chunks_size = meta.chunks_size
    model = load_model()
    _prepare_meta(repo_id, out_dir)

    total = len(episodes)
    t0 = time.time()
    total_frames = 0
    shard_stats = []

    for i, episode_index in enumerate(episodes):
        chunk = episode_index // chunks_size
        rel_path = f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        src_parquet = _download_meta_file(repo_id, rel_path)
        df = pd.read_parquet(src_parquet)

        obs = np.stack(df["observation.state"].to_numpy()).astype(float)
        result = compute_obs_ee(model, obs)
        left_ee = result["left_ee"].astype(np.float32)
        right_ee = result["right_ee"].astype(np.float32)
        df["observation.left_ee"] = list(left_ee)
        df["observation.right_ee"] = list(right_ee)

        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)

        shard_stats.append({
            "episode_index": episode_index,
            "left_ee_stats": _stats_for(left_ee),
            "right_ee_stats": _stats_for(right_ee),
        })

        total_frames += len(df)

        if (i + 1) % 25 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - (i + 1)) / rate if rate > 0 else float("nan")
            print(
                f"[shard {shard_label}] [{i + 1}/{total}] episode {episode_index} ({len(df)} frames) — "
                f"{elapsed:6.0f}s elapsed, ~{eta:6.0f}s left, {total_frames} frames so far"
            )

    partial_path = out_dir / "meta" / f"episodes_stats.partial.{shard_label}.jsonl"
    partial_path.write_text("\n".join(json.dumps(r) for r in shard_stats) + ("\n" if shard_stats else ""))

    print(f"[shard {shard_label}] Done. Wrote {len(episodes)} episodes to {out_dir}")
    print(f"[shard {shard_label}] {total_frames} frames FK'd.")


def merge_stats(repo_id: str, out_dir: Path) -> None:
    """Combines every meta/episodes_stats.partial.*.jsonl shard output with
    the untouched source stats into the final meta/episodes_stats.jsonl, and
    writes the final meta/info.json. Run once after all shards finish."""
    info = _prepare_meta(repo_id, out_dir)

    stats_path = _download_meta_file(repo_id, "meta/episodes_stats.jsonl")
    stats_lines = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    stats_by_episode = {rec["episode_index"]: rec for rec in stats_lines}

    partial_files = sorted((out_dir / "meta").glob("episodes_stats.partial.*.jsonl"))
    n_merged = 0
    for pf in partial_files:
        for line in pf.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            target = stats_by_episode.get(rec["episode_index"])
            if target is not None:
                target["stats"]["observation.left_ee"] = rec["left_ee_stats"]
                target["stats"]["observation.right_ee"] = rec["right_ee_stats"]
                n_merged += 1

    missing = [
        idx for idx, rec in stats_by_episode.items()
        if "observation.left_ee" not in rec["stats"] or "observation.right_ee" not in rec["stats"]
    ]

    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (out_dir / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(stats_by_episode[idx]) for idx in sorted(stats_by_episode)) + "\n"
    )
    print(f"Merged {n_merged} episode stats from {len(partial_files)} shard file(s).")
    if missing:
        print(f"WARNING: {len(missing)} episodes have no observation.{{left,right}}_ee stats yet (shards still pending?): "
              f"{missing[:20]}{'...' if len(missing) > 20 else ''}")
    else:
        print("All episodes accounted for.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, default=REPO_ID_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "output" / "icl-dataset-obs-ee")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--shard", type=int, default=None, help="process episodes where episode_index %% num_shards == shard")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-stats", action="store_true", help="combine all shards' partial stats into the final meta/episodes_stats.jsonl")
    args = parser.parse_args()

    if args.merge_stats:
        merge_stats(args.repo_id, args.out)
        return

    meta = get_metadata(args.repo_id)
    total_episodes = meta.total_episodes

    if args.shard is not None:
        episodes = [i for i in range(total_episodes) if i % args.num_shards == args.shard]
        shard_label = f"{args.shard}-of-{args.num_shards}"
    else:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else total_episodes
        episodes = list(range(start, end))
        shard_label = f"{start}-{end}"

    generate(args.repo_id, args.out, episodes, shard_label)


if __name__ == "__main__":
    main()
