"""Reconstruct joint-space q_targets from the cartesian action states in
adityx23/icl-dataset, following the mink-based IK procedure documented in
vr_teleop_ik.md (sections 4, 5, 6, 9, 10). Streams one episode at a time via
the `lerobot` package's LeRobotDataset (no bulk download, no videos).

Run inside the `lerobot` conda env: `conda activate lerobot`.

Usage:
    python convert_actions_to_joints.py --episode 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ik_pipeline import REPO_ID_DEFAULT, compute_action_q, load_episode, load_model


def convert_episode(repo_id: str, episode_index: int, out_path: Path) -> None:
    ep = load_episode(repo_id, episode_index)
    n = ep["observation_state"].shape[0]
    print(f"Loaded episode {episode_index}: {n} rows (task={ep['task']!r}, success={ep['success']})")

    model = load_model()
    result = compute_action_q(model, ep["observation_state"], ep["action_left_ee"], ep["action_right_ee"])

    obs = ep["observation_state"]
    rows = {
        "timestamp": ep["timestamp"],
        "frame_index": np.arange(n),
        "episode_uid": ep["episode_uid"],
        **{f"obs_lj{j}": obs[:, j] for j in range(7)},
        **{f"obs_rj{j}": obs[:, 7 + j] for j in range(7)},
        "obs_lift": obs[:, 14],
        **{f"qtarget_lj{j}": result["qtarget_left"][:, j] for j in range(7)},
        **{f"qtarget_rj{j}": result["qtarget_right"][:, j] for j in range(7)},
        "left_solved": result["left_solved"],
        "left_pos_err": result["left_pos_err"],
        "left_rot_err": result["left_rot_err"],
        "right_solved": result["right_solved"],
        "right_pos_err": result["right_pos_err"],
        "right_rot_err": result["right_rot_err"],
    }
    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"Wrote {len(out_df)} rows to {out_path}")

    for arm in ("left", "right"):
        solved = result[f"{arm}_solved"]
        pos_err = result[f"{arm}_pos_err"]
        rot_err = result[f"{arm}_rot_err"]
        print(
            f"{arm:>5} arm: solved {solved.mean() * 100:5.1f}% of frames | "
            f"pos_err mean/max = {pos_err.mean():.2e}/{pos_err.max():.2e} m | "
            f"rot_err mean/max = {rot_err.mean():.2e}/{rot_err.max():.2e} rad"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--repo-id", type=str, default=REPO_ID_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or Path(__file__).parent / "output" / f"episode_{args.episode:06d}_qtargets.parquet"
    convert_episode(args.repo_id, args.episode, out_path)


if __name__ == "__main__":
    main()
