# ICL IK Visualizer

Streams episodes from a `lerobot` v2.1-format dataset
([`adityx23/icl-dataset`](https://huggingface.co/datasets/adityx23/icl-dataset))
and reconstructs joint-space `q_target`s from the recorded cartesian
`action.left_ee`/`action.right_ee` states via a per-frame, warm-started
mink IK solve (`ik_pipeline.py`).

## Setup

```powershell
conda activate lerobot   # already has lerobot, huggingface_hub, fastapi, uvicorn
pip install mink mujoco  # added on top, for the IK solve
```

## Running

```powershell
.\start_server.ps1
.\start_server.ps1 -Dataset "owner/other-dataset" -Port 9000
```

or directly: `python server.py --dataset adityx23/icl-dataset --port 8010`.

Open the printed URL, pick an episode, and the recorded joint angles plot
immediately (no bulk dataset/video download). "Calculate action q" runs the
IK reconstruction for that episode and caches the result to
`output/action_q_cache/`.

## Regenerating the full dataset

`generate_full_dataset.py` reprocesses every episode and writes a `parquet`
with an added `action.q_target` column, sharded by
`episode_index % num_shards` for balanced parallel runs (episode length
varies a lot):

```powershell
python generate_full_dataset.py --shard 0 --num-shards 6   # run once per shard
python generate_full_dataset.py --merge-stats --num-shards 6   # after all shards finish
```

## Files

```
ik_pipeline.py                shared: episode streaming + mink IK solve
convert_actions_to_joints.py  CLI wrapper around ik_pipeline (single episode)
generate_full_dataset.py      full-dataset regeneration, sharded
server.py                     FastAPI backend for the visualizer
static/                       frontend (index.html, app.js, style.css)
assets/yor_kin.mjcf           kinematics-only MJCF (meshes stripped)
output/                       parquet/cache written by the CLI and server (git-ignored)
```

## Known data quirk: orientation glitches

`action.{left,right}_ee`'s recorded orientation occasionally snaps by
90-180 degrees for a frame or two — a source-data artifact (confirmed
against `observation.state`, which stays smooth through the same frames),
not a real robot event. Left alone, one glitch derails the sequential
warm-started solver for thousands of frames afterward, since it has no
other way to recover a sane pose. `ik_pipeline.py` guards against this two
ways:

- `_declitch_quaternions` flags any frame-to-frame rotation exceeding a
  physically implausible angular velocity and SLERPs over it.
- `ArmIK.reground`, triggered after `RESEED_AFTER_UNSOLVED` consecutive
  unsolved frames, hard-resets the solver's warm start to the real recorded
  joint state so any stall this misses is self-limiting instead of
  cascading for the rest of the episode.
