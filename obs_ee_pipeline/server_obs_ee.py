"""Backend for the observation-FK -> end-effector-pose visualizer.

Streams episodes from adityx23/icl-dataset the same way server.py does (see
that file's docstring for the caching rationale) and, on every episode load,
forward-kinematics that episode's observation.state into per-arm end-effector
poses via ik_pipeline.compute_obs_ee (vr_teleop_ik.md §9) -- the FK
counterpart of server.py's IK reconstruction. Unlike the IK path there is no
iterative solve, so this is cheap enough to run synchronously on every
request; no on-disk cache, no separate "calculate" step.

Run inside the `lerobot` conda env: `conda activate lerobot`.
"""

import argparse
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from common.ik_pipeline import REPO_ID_DEFAULT, align_action_to_obs, compute_obs_ee, get_metadata, get_video_urls, load_episode, load_model

STATIC_DIR = BASE_DIR / "static"


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Obs->EE FK visualizer server", add_help=False)
    parser.add_argument("--dataset", "--repo-id", dest="dataset", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("-h", "--help", action="help")
    args, _unknown = parser.parse_known_args()
    return args


_CLI_ARGS = _parse_cli_args()
REPO_ID = _CLI_ARGS.dataset or os.environ.get("ICL_DATASET_REPO_ID") or REPO_ID_DEFAULT

print(f"Loading {REPO_ID} metadata...")
_META = get_metadata(REPO_ID)
TOTAL_EPISODES = _META.total_episodes

_model = load_model()

# Same arm-prefixed flattening for both source dict order (EE_POSE_NAMES_QUAT
# = qw,qx,qy,qz,x,y,z) so the frontend can key straight into a flat dict, same
# pattern as server.py's JOINT_LABELS.
_ARM_LABELS = {
    "left": [f"left_{n}" for n in ["qw", "qx", "qy", "qz", "x", "y", "z"]],
    "right": [f"right_{n}" for n in ["qw", "qx", "qy", "qz", "x", "y", "z"]],
}

_TASK_INDEX_BY_EPISODE: dict[int, int] = {}
_EPISODES_BY_TASK: dict[int, list[dict]] = {}
for _idx, _ep in _META.episodes.items():
    _t_idx = _META.task_to_task_index[_ep["tasks"][0]]
    _TASK_INDEX_BY_EPISODE[_idx] = _t_idx
    _EPISODES_BY_TASK.setdefault(_t_idx, []).append(
        {"episode_index": _idx, "length": _ep["length"], "success": _ep.get("success")}
    )
for _eps in _EPISODES_BY_TASK.values():
    _eps.sort(key=lambda e: e["episode_index"])

app = FastAPI(title="ICL Obs->EE FK Visualizer")


@app.get("/api/info")
def get_info():
    return {"repo_id": REPO_ID, "total_episodes": TOTAL_EPISODES, "fps": _META.fps}


@app.get("/api/tasks")
def list_tasks():
    out = [
        {"task_index": t_idx, "task": _META.tasks[t_idx], "total_episodes": len(eps)}
        for t_idx, eps in _EPISODES_BY_TASK.items()
    ]
    out.sort(key=lambda t: t["task"].lower())
    return out


@app.get("/api/tasks/{task_index}/episodes")
def list_task_episodes(task_index: int):
    eps = _EPISODES_BY_TASK.get(task_index)
    if eps is None:
        raise HTTPException(404, "task not found")
    return [{"position": i + 1, **e} for i, e in enumerate(eps)]


@app.get("/api/episode/{episode_index}")
def get_episode(episode_index: int):
    if not (0 <= episode_index < TOTAL_EPISODES):
        raise HTTPException(404, f"episode_index out of range [0, {TOTAL_EPISODES})")
    ep = load_episode(REPO_ID, episode_index)

    fk = compute_obs_ee(_model, ep["observation_state"])  # {"left_ee": (N,7), "right_ee": (N,7)}

    obs = {}
    action = {}
    for arm, action_key in (("left", "action_left_ee"), ("right", "action_right_ee")):
        fk_arr = fk[f"{arm}_ee"]
        corrected_action = align_action_to_obs(fk_arr, ep[action_key])
        for i, label in enumerate(_ARM_LABELS[arm]):
            obs[label] = fk_arr[:, i].tolist()
            action[label] = corrected_action[:, i].tolist()

    task_index = _TASK_INDEX_BY_EPISODE[episode_index]
    task_episodes = _EPISODES_BY_TASK[task_index]
    position = next(i + 1 for i, e in enumerate(task_episodes) if e["episode_index"] == episode_index)

    return {
        "episode_index": episode_index,
        "total_episodes": TOTAL_EPISODES,
        "task": ep["task"],
        "task_index": task_index,
        "position": position,
        "total_in_task": len(task_episodes),
        "success": ep["success"],
        "length": ep["length"],
        "fps": ep["fps"],
        "episode_uid": ep["episode_uid"],
        "timestamp": ep["timestamp"].tolist(),
        "obs": obs,
        "action": action,
        "videos": get_video_urls(REPO_ID, episode_index),
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_CLI_ARGS.host, port=_CLI_ARGS.port)
