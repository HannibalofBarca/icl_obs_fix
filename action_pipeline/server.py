"""Backend for the cartesian-action -> joint-target IK visualizer.

Streams episodes from adityx23/icl-dataset via the `lerobot` package's
LeRobotDatasetMetadata (cached once at startup, see ik_pipeline.get_metadata)
plus a single per-episode parquet fetch — no bulk download. Videos are never
downloaded either: the frontend gets direct Hub `resolve` URLs and streams
them itself. On demand, runs the mink-based IK reconstruction described in
vr_teleop_ik.md to turn each frame's cartesian action into a joint target.

Run inside the `lerobot` conda env: `conda activate lerobot`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.ik_pipeline import REPO_ID_DEFAULT, compute_action_q, get_metadata, get_video_urls, load_episode, load_model

STATIC_DIR = BASE_DIR / "static"
CACHE_DIR = PROJECT_ROOT / "output" / "action_q_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="IK visualizer server", add_help=False)
    parser.add_argument("--dataset", "--repo-id", dest="dataset", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("-h", "--help", action="help")
    args, _unknown = parser.parse_known_args()
    return args


_CLI_ARGS = _parse_cli_args()
REPO_ID = _CLI_ARGS.dataset or os.environ.get("ICL_DATASET_REPO_ID") or REPO_ID_DEFAULT

# Paid once here, at startup, not on the first (or every) API request — see
# get_metadata's docstring for why a fresh LeRobotDatasetMetadata/LeRobotDataset
# per request was the actual cause of the visualizer being slow to load.
print(f"Loading {REPO_ID} metadata...")
_META = get_metadata(REPO_ID)
TOTAL_EPISODES = _META.total_episodes

JOINT_LABELS = [f"lj{i}" for i in range(7)] + [f"rj{i}" for i in range(7)] + ["lift"]

_model = load_model()

# episode_index -> task_index, and task_index -> ordered list of that task's
# episodes, both built once here (mirrors Manual Annotation/server.py's
# TASKS_BY_INDEX) rather than recomputed per request.
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

app = FastAPI(title="ICL IK Visualizer")


def _cache_path(episode_index: int) -> Path:
    return CACHE_DIR / f"episode_{episode_index:06d}.json"


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
    obs = ep["observation_state"]

    cached = None
    cp = _cache_path(episode_index)
    if cp.exists():
        cached = json.loads(cp.read_text())

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
        "obs": {label: obs[:, i].tolist() for i, label in enumerate(JOINT_LABELS)},
        "action_q": cached,
        "videos": get_video_urls(REPO_ID, episode_index),
    }


@app.post("/api/episode/{episode_index}/action-q")
def calculate_action_q(episode_index: int, force: bool = False):
    if not (0 <= episode_index < TOTAL_EPISODES):
        raise HTTPException(404, f"episode_index out of range [0, {TOTAL_EPISODES})")

    cp = _cache_path(episode_index)
    if cp.exists() and not force:
        return json.loads(cp.read_text())

    ep = load_episode(REPO_ID, episode_index)
    result = compute_action_q(_model, ep["observation_state"], ep["action_left_ee"], ep["action_right_ee"], ep["timestamp"])

    qtarget = np.concatenate([result["qtarget_left"], result["qtarget_right"]], axis=1)  # (N, 14)
    payload = {
        "episode_index": episode_index,
        "qtarget": {label: qtarget[:, i].tolist() for i, label in enumerate(JOINT_LABELS[:14])},
        "diagnostics": {
            "left": {
                "solved_rate": float(np.mean(result["left_solved"])),
                "pos_err_mean": float(np.mean(result["left_pos_err"])),
                "pos_err_max": float(np.max(result["left_pos_err"])),
                "rot_err_mean": float(np.mean(result["left_rot_err"])),
                "rot_err_max": float(np.max(result["left_rot_err"])),
            },
            "right": {
                "solved_rate": float(np.mean(result["right_solved"])),
                "pos_err_mean": float(np.mean(result["right_pos_err"])),
                "pos_err_max": float(np.max(result["right_pos_err"])),
                "rot_err_mean": float(np.mean(result["right_rot_err"])),
                "rot_err_max": float(np.max(result["right_rot_err"])),
            },
        },
    }
    cp.write_text(json.dumps(payload))
    return payload


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_CLI_ARGS.host, port=_CLI_ARGS.port)
