# ICL IK Visualizer

Two visualizers/pipelines over a `lerobot` v2.1-format dataset
([`adityx23/icl-dataset`](https://huggingface.co/datasets/adityx23/icl-dataset)),
both built on the shared episode streaming + mink IK/FK code in
`common/ik_pipeline.py`:

- **`action_pipeline/`** — reconstructs joint-space `q_target`s from the
  recorded cartesian `action.left_ee`/`action.right_ee` states via a
  per-frame, warm-started mink IK solve.
- **`obs_ee_pipeline/`** — forward-kinematics the recorded
  `observation.state` (real joint encoders) into per-arm end-effector poses;
  the FK counterpart of the above, cheap enough to run synchronously with no
  cache.

## Setup

```powershell
conda activate lerobot   # already has lerobot, huggingface_hub, fastapi, uvicorn
pip install mink mujoco  # added on top, for the IK/FK solve
```

## Running

```powershell
.\action_pipeline\start_server.ps1
.\action_pipeline\start_server.ps1 -Dataset "owner/other-dataset" -Port 9000

.\obs_ee_pipeline\start_server.ps1
.\obs_ee_pipeline\start_server.ps1 -Dataset "owner/other-dataset" -Port 9020
```

or directly: `python action_pipeline/server.py --dataset adityx23/icl-dataset --port 8010`
/ `python obs_ee_pipeline/server_obs_ee.py --dataset adityx23/icl-dataset --port 8020`.

Open the printed URL, pick an episode, and the recorded joint angles plot
immediately (no bulk dataset/video download). In the action-pipeline
visualizer, "Calculate action q" runs the IK reconstruction for that episode
and caches the result to `output/action_q_cache/`.

## Regenerating the full dataset

`action_pipeline/generate_full_dataset.py` and
`obs_ee_pipeline/generate_obs_ee_dataset.py` reprocess every episode and
write a `parquet` with the added columns (`action.q_target`, or
`observation.{left,right}_ee`), sharded by `episode_index % num_shards` for
balanced parallel runs (episode length varies a lot):

```powershell
python action_pipeline/generate_full_dataset.py --shard 0 --num-shards 6
python action_pipeline/generate_full_dataset.py --merge-stats --num-shards 6

python obs_ee_pipeline/generate_obs_ee_dataset.py --shard 0 --num-shards 6
python obs_ee_pipeline/generate_obs_ee_dataset.py --merge-stats --num-shards 6
```

## Files

```
common/ik_pipeline.py                        shared: episode streaming + mink IK/FK solve
common/assets/yor_kin.mjcf                    kinematics-only MJCF (meshes stripped)

action_pipeline/convert_actions_to_joints.py  CLI wrapper around ik_pipeline (single episode)
action_pipeline/generate_full_dataset.py      full-dataset regeneration, sharded
action_pipeline/server.py                     FastAPI backend for the IK visualizer
action_pipeline/static/                       frontend (index.html, app.js, style.css)
action_pipeline/start_server.ps1

obs_ee_pipeline/convert_obs_to_ee.py          CLI wrapper around ik_pipeline (single episode)
obs_ee_pipeline/generate_obs_ee_dataset.py    full-dataset regeneration, sharded
obs_ee_pipeline/server_obs_ee.py              FastAPI backend for the obs->EE FK visualizer
obs_ee_pipeline/static/                       frontend (index.html, app.js, style.css)
obs_ee_pipeline/start_server.ps1

output/                                       parquet/cache written by the CLIs and servers (git-ignored)
logs/                                         run logs from past shard regenerations (git-ignored)
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
