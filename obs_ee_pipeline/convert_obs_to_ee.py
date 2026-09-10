"""Forward-kinematics the recorded observation.state (real joint encoders)
in adityx23/icl-dataset into per-arm end-effector poses, via the FK
counterpart to convert_actions_to_joints.py's IK reconstruction (see
ik_pipeline.compute_obs_ee and vr_teleop_ik.md §9). Streams one episode at a
time via the `lerobot` package's LeRobotDataset (no bulk download, no
videos).

Run inside the `lerobot` conda env: `conda activate lerobot`.

Usage:
    python convert_obs_to_ee.py --episode 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.ik_pipeline import EE_POSE_NAMES, REPO_ID_DEFAULT, compute_obs_ee, load_episode, load_model


def convert_episode(repo_id: str, episode_index: int, out_path: Path) -> None:
    ep = load_episode(repo_id, episode_index)
    n = ep["observation_state"].shape[0]
    print(f"Loaded episode {episode_index}: {n} rows (task={ep['task']!r}, success={ep['success']})")

    model = load_model()
    result = compute_obs_ee(model, ep["observation_state"])

    obs = ep["observation_state"]
    rows = {
        "timestamp": ep["timestamp"],
        "frame_index": np.arange(n),
        "episode_uid": ep["episode_uid"],
        **{f"obs_lj{j}": obs[:, j] for j in range(7)},
        **{f"obs_rj{j}": obs[:, 7 + j] for j in range(7)},
        "obs_lift": obs[:, 14],
        **{f"obs_left_ee_{name}": result["left_ee"][:, k] for k, name in enumerate(EE_POSE_NAMES)},
        **{f"obs_right_ee_{name}": result["right_ee"][:, k] for k, name in enumerate(EE_POSE_NAMES)},
    }
    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"Wrote {len(out_df)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--repo-id", type=str, default=REPO_ID_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or Path(__file__).resolve().parent.parent / "output" / f"episode_{args.episode:06d}_obs_ee.parquet"
    convert_episode(args.repo_id, args.episode, out_path)


if __name__ == "__main__":
    main()
