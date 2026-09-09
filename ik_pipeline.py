"""Shared pipeline: stream an episode from adityx23/icl-dataset via the
`lerobot` package's LeRobotDataset, and reconstruct joint-space q_targets
from its cartesian action states per vr_teleop_ik.md (sections 4, 5, 6, 9, 10).

Meant to run inside the `lerobot` conda env (has `lerobot`, `huggingface_hub`,
plus `mink`/`mujoco` installed on top for the IK solve).
"""

from pathlib import Path, PurePosixPath

import mink
import mujoco
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, hf_hub_url

REPO_ID_DEFAULT = "adityx23/icl-dataset"
MJCF_PATH = Path(__file__).parent / "assets" / "yor_kin.mjcf"

# adityx23/icl-dataset carries no Hub revision tag (e.g. "v2.1"), so anything
# resolved through lerobot's own version logic (LeRobotDatasetMetadata.revision)
# is unreliable — it's only "main" if that instance happened to hit the
# patched except-branch download, and stays at the "v2.1" default otherwise
# (whenever meta/ was already cached locally). Every direct download/URL this
# module makes pins "main" explicitly instead of trusting that attribute.
HUB_REVISION = "main"

# Solver config, exactly as vr_teleop_ik.md §6 / §10.4.
SOLVER_DT = 0.01
MAX_ITER = 10
POS_EPS = 1e-3
ROT_EPS = 1e-3
DAMPING = 1e-5
POSITION_COST = 1.0
ORIENTATION_COST = 0.1
LM_DAMPING = 1.0
LIFT_POSTURE_COST = 1e-1
ARM_POSTURE_COST = 1e-3

# `action.{left,right}_ee`'s recorded orientation occasionally snaps by
# 90-180 degrees in a single frame and decays back over the next few frames,
# while position stays continuous and `observation.state` (the real robot's
# encoders) shows no jump at all -- confirmed a source-data artifact, not a
# real robot event. 720 deg/s is a generous physical cap for a human-driven
# teleop wrist (sampled non-glitched episodes topped out around 220 deg/s)
# while sitting far below every observed glitch (600-5400 deg/s), so it
# separates the two cleanly. See _declitch_quaternions.
MAX_ANGULAR_VELOCITY_DEG_S = 720.0

# Safety net for any spike _declitch_quaternions doesn't catch: ArmIK.solve
# warm-starts every frame from the previous frame's result, so one bad target
# can shove the solver into a joint-limited configuration it then has no way
# to recover from -- observed taking >1800 frames to wander back to a clean
# solve on its own. Force a hard re-seed from the real observation.state
# after this many consecutive unsolved frames so a stall is self-limiting.
RESEED_AFTER_UNSOLVED = 15


_METADATA_CACHE: dict = {}


def get_metadata(repo_id: str):
    """LeRobotDatasetMetadata, built once per repo_id and reused for the rest
    of the process's life (same pattern as Data Annotation/viz_server.py's
    get_metadata) — this is metadata only (episodes.jsonl/info.json/etc, no
    per-episode data or video), so it's cheap to keep around and expensive to
    rebuild: every fresh LeRobotDatasetMetadata()/LeRobotDataset() re-parses
    all ~3k episodes' worth of episodes.jsonl, and — the actual cause of the
    multi-second-per-click slowness this replaced — instantiating a full
    LeRobotDataset per request always re-triggers its except-branch
    `snapshot_download` network round trip, because its own file-existence
    check unconditionally includes this dataset's video files regardless of
    download_videos=False, which are never actually downloaded here."""
    if repo_id not in _METADATA_CACHE:
        import lerobot.datasets.lerobot_dataset as lrd

        # lerobot 0.3.3's version resolution unconditionally raises on any
        # dataset with no codebase version tag on the Hub (this one
        # included), compounded by a huggingface_hub>=1.0 exception-signature
        # mismatch in that same lerobot version. Patch it to just accept
        # "main" — every other code path (minimal per-file fetch, no bulk
        # snapshot) is unaffected.
        lrd.get_safe_version = lambda repo_id, revision: "main"
        _METADATA_CACHE[repo_id] = lrd.LeRobotDatasetMetadata(repo_id)
    return _METADATA_CACHE[repo_id]


def load_episode(repo_id: str, episode_index: int) -> dict:
    """Streams just this one episode's parquet — a single hf_hub_download
    call (near-instant once cached locally, since it's an ETag check against
    one known file, not a repo-tree snapshot) plus a plain pandas parquet
    read. No videos, no bulk download, no full LeRobotDataset construction."""
    meta = get_metadata(repo_id)
    ep_meta = meta.episodes[episode_index]

    rel_path = PurePosixPath(meta.get_data_file_path(episode_index)).as_posix()
    parquet_path = hf_hub_download(repo_id, rel_path, repo_type="dataset", revision=HUB_REVISION)
    df = pd.read_parquet(parquet_path)

    def col(name):
        return np.stack(df[name].to_numpy())

    return {
        "episode_index": episode_index,
        "task": ep_meta["tasks"][0],
        "success": ep_meta.get("success"),
        "length": ep_meta["length"],
        "episode_uid": ep_meta.get("episode_uid"),
        "fps": meta.fps,
        "timestamp": df["timestamp"].to_numpy(dtype=float),
        "observation_state": col("observation.state").astype(float),  # (N, 15)
        "action_left_ee": col("action.left_ee").astype(float),  # (N, 7) qw,qx,qy,qz,x,y,z
        "action_right_ee": col("action.right_ee").astype(float),  # (N, 7)
        "action_gripper": col("action.gripper").astype(float),  # (N, 2)
        "action_base_vel": col("action.base_vel").astype(float),  # (N, 3)
        "action_lift_cmd": df["action.lift_cmd"].apply(lambda v: v[0]).to_numpy(dtype=float),  # (N,)
        "valid": bool(df["valid"].iloc[0]) if len(df) else None,
        "keep": bool(df["keep"].iloc[0]) if len(df) else None,
    }


def get_video_urls(repo_id: str, episode_index: int) -> dict:
    """Direct Hub `resolve` URLs for this episode's cameras — no download,
    ever (same technique as Data Annotation/web_visualizer.py's
    build_episode_payload): the Hub serves these with CORS + byte-range
    support, so a <video> tag streams/scrubs straight from it."""
    meta = get_metadata(repo_id)
    urls = {}
    for camera_key in meta.video_keys:
        rel_path = PurePosixPath(meta.get_video_file_path(episode_index, camera_key)).as_posix()
        urls[camera_key] = hf_hub_url(repo_id, rel_path, repo_type="dataset", revision=HUB_REVISION)
    return urls


class ArmIK:
    """Mirrors SingleArmIK from vr_teleop_ik.md §6, built on one shared model."""

    def __init__(self, model: mujoco.MjModel, arm: str, lift_dof: int):
        self.model = model
        self.arm = arm
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_arm_joint{i}")
            for i in range(1, 8)
        ]
        self.qpos_ids = model.jnt_qposadr[joint_ids]

        self.configuration = mink.Configuration(model)

        cost = np.full(model.nv, ARM_POSTURE_COST)
        cost[lift_dof] = LIFT_POSTURE_COST
        self.posture_task = mink.PostureTask(model, cost=cost)

        self.frame_task = mink.FrameTask(
            frame_name=f"{arm}_arm_ee",
            frame_type="site",
            position_cost=POSITION_COST,
            orientation_cost=ORIENTATION_COST,
            lm_damping=LM_DAMPING,
        )
        self.tasks = [self.frame_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(model)]
        self._initialized = False

    def init(self, q_full: np.ndarray) -> None:
        """Seed once per episode from the recorded joint state (§4, §10.2)."""
        self.configuration.update(q_full)
        self.posture_task.set_target_from_configuration(self.configuration)
        self._initialized = True

    def solve(self, quat_wxyz: np.ndarray, pos_xyz: np.ndarray):
        """One IK solve, warm-started from whatever self.configuration already holds."""
        assert self._initialized, "call init() once before the first solve()"
        target = mink.SE3.from_rotation_and_translation(mink.SO3(quat_wxyz), pos_xyz)
        self.frame_task.set_target(target)

        is_solved = False
        err = np.zeros(6)
        for _ in range(MAX_ITER):
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                SOLVER_DT,
                solver="daqp",
                damping=DAMPING,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, SOLVER_DT)
            err = self.frame_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) <= POS_EPS and np.linalg.norm(err[3:]) <= ROT_EPS:
                is_solved = True
                break

        q_arm = self.configuration.q[self.qpos_ids].copy()
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        return q_arm, is_solved, pos_err, rot_err

    def reground(self, q_full: np.ndarray) -> None:
        """Hard-resets the warm start to the real recorded joint state
        (leaving the posture task's nullspace target as-is, still frame 0's
        pose) -- a mid-episode stall recovery, see RESEED_AFTER_UNSOLVED."""
        self.configuration.update(q_full)


def build_full_qpos(model: mujoco.MjModel, obs_state: np.ndarray) -> np.ndarray:
    """observation.state is [lj0..lj6, rj0..rj6, lift] -> full 15-dim qpos."""
    q = np.zeros(model.nq)
    left_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_arm_joint{i}") for i in range(1, 8)]
    right_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_arm_joint{i}") for i in range(1, 8)]
    lift_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift_joint")
    q[model.jnt_qposadr[left_ids]] = obs_state[0:7]
    q[model.jnt_qposadr[right_ids]] = obs_state[7:14]
    q[model.jnt_qposadr[lift_id]] = obs_state[14]
    return q


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MJCF_PATH))


# Column-name convention for a 7-dim SE3 pose, matching action.left_ee /
# action.right_ee's own feature schema in the source dataset exactly
# (dtype=float32, shape=[7], names=[...]) so observation.left_ee /
# observation.right_ee slot in as directly comparable columns.
EE_POSE_NAMES = ["qw", "qx", "qy", "qz", "x", "y", "z"]


def compute_obs_ee(model: mujoco.MjModel, observation_state: np.ndarray) -> dict:
    """Forward-kinematics every frame's recorded observation.state (real
    joint encoders) through the MJCF (§9) to get each arm's end-effector
    pose -- the FK counterpart to compute_action_q's IK. observation_state
    is fully determined qpos already, so computing this FK itself needs no
    iterative solve, no warm-starting, and no anchor correction.

    That said, the *comparison* of this output against the raw recorded
    action.{left,right}_ee still needs the same per-episode/per-arm SE3 bias
    removed that compute_action_q corrects for on the IK side (§5's
    "Important caveat") -- see align_action_to_obs, immediately below
    _anchor_correction/_corrected_target further down this file.

    observation_state: (N, 15) -> {"left_ee": (N, 7), "right_ee": (N, 7)},
    each row qw,qx,qy,qz,x,y,z (matches EE_POSE_NAMES / action.{left,right}_ee's
    own column convention)."""
    n = observation_state.shape[0]
    configuration = mink.Configuration(model)

    left_ee = np.zeros((n, 7))
    right_ee = np.zeros((n, 7))
    for i in range(n):
        configuration.update(build_full_qpos(model, observation_state[i]))
        left_T = configuration.get_transform_frame_to_world("left_arm_ee", "site")
        right_T = configuration.get_transform_frame_to_world("right_arm_ee", "site")
        left_ee[i, 0:4] = left_T.rotation().wxyz
        left_ee[i, 4:7] = left_T.translation()
        right_ee[i, 0:4] = right_T.rotation().wxyz
        right_ee[i, 4:7] = right_T.translation()

    # The rotation-matrix -> quaternion conversion behind get_transform_frame_to_world
    # (ultimately mujoco's mju_mat2Quat) picks *a* valid quaternion for each frame's
    # matrix independently, with no guarantee of picking the same sign/hemisphere as
    # the previous frame's -- quaternions doubly cover SO(3), so q and -q represent
    # the identical rotation. A frame landing on the opposite hemisphere from its
    # neighbor is a genuinely continuous rotation but reads as a sign-flip
    # discontinuity in every raw component at once (observed directly: consecutive-
    # frame dot product of exactly -1). This is purely a representation artifact of
    # computing each frame independently, not a data glitch -- unlike
    # _declitch_quaternions further down, which repairs actual bad recorded values.
    left_ee[:, 0:4] = _enforce_quaternion_hemisphere(left_ee[:, 0:4])
    right_ee[:, 0:4] = _enforce_quaternion_hemisphere(right_ee[:, 0:4])

    return {"left_ee": left_ee, "right_ee": right_ee}


def _enforce_quaternion_hemisphere(quat_wxyz: np.ndarray) -> np.ndarray:
    """Flips each frame's quaternion sign to match the previous frame's
    hemisphere whenever their dot product is negative -- q and -q represent
    the same rotation, so this changes nothing physically, it just picks the
    continuously-varying representative so a time series of quaternions from
    independently-computed frames doesn't show a spurious all-components sign
    flip. See compute_obs_ee's comment for why this is needed there."""
    out = quat_wxyz.copy()
    for i in range(1, out.shape[0]):
        if np.dot(out[i - 1], out[i]) < 0:
            out[i] = -out[i]
    return out


def _anchor_correction(seed_transform: "mink.SE3", first_quat_wxyz: np.ndarray, first_pos_xyz: np.ndarray) -> "mink.SE3":
    """Per-episode, per-arm rigid correction for a real calibration offset
    found in the source data: `action.left_ee`/`action.right_ee` targets
    carry a session-specific, near-constant SE3 bias relative to this MJCF's
    site frame (confirmed empirically — almost pure translation, ~0 in X/Y,
    consistently tens of cm in Z, stable across an entire episode/session but
    different between arms and between sessions/tasks — e.g. "assemble the
    two gears" measures left≈-0.185m/right≈-0.037m while "open the gatorade
    bottle" measures left≈-0.14m/right≈-0.215m, both already present at frame
    0 before any motion). Likely a per-session tool/gripper (or similar)
    offset baked into the live system's own `X_ee_init` clutch anchor (§5)
    that this MJCF's fixed site placement doesn't reproduce for every
    session. Uncorrected, this can push the redundant elbow/wrist joints to
    track a systematically displaced target for the whole episode, in the
    worst cases hard enough to run into a mechanical joint limit and stall
    convergence (seen directly: one episode's right arm dropped to a 10%
    solved-rate with the elbow pinned at its limit).

    Fix: anchor frame 0's *recorded* target to that frame's *actual* achieved
    pose (this episode's own seed, which is ground truth — §4) and apply the
    same rigid transform to every subsequent frame's target. This assumes
    the bias is a fixed SE3 for the whole episode, which is exactly what the
    per-session consistency above shows."""
    first_target = mink.SE3.from_rotation_and_translation(mink.SO3(first_quat_wxyz), first_pos_xyz)
    return seed_transform.multiply(first_target.inverse())


def _declitch_quaternions(quat_wxyz: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
    """Repairs the recorded-orientation spikes described at
    MAX_ANGULAR_VELOCITY_DEG_S's definition: flags any rotation from the
    last confirmed-good frame exceeding that angular velocity, then replaces
    each contiguous flagged run by SLERPing between the nearest good
    quaternions bracketing it (held constant off the nearest good frame if
    the run touches either end of the episode).

    Deliberately compares each candidate frame to the last *confirmed-good*
    frame rather than the immediately preceding raw sample: a glitch has
    been observed held/duplicated verbatim across two or more consecutive
    raw samples, which reads as zero velocity against the immediately
    preceding (also-bad) sample and would otherwise slip through undetected
    -- e.g. frame i snaps to a bad orientation, frame i+1 repeats that same
    bad value bit-for-bit, so a prev-frame-only check flags i but not i+1,
    and that lone undeclitched bad target is enough to derail a downstream
    sequential warm-started IK solve for many frames afterward. Comparing
    against the last known-good frame instead naturally propagates the flag
    through any such held-duplicate run, since elapsed time accumulates
    across it and a real (non-glitch) resumption of motion still measures a
    physically plausible velocity once it's checked against that same
    anchor."""
    n = quat_wxyz.shape[0]
    out = quat_wxyz.copy()
    if n < 2:
        return out

    dt = np.diff(timestamp)
    good_dt = dt[dt > 1e-6]
    dt = np.where(dt > 1e-6, dt, np.median(good_dt) if good_dt.size else 1.0)

    bad = np.zeros(n, dtype=bool)
    last_good = 0
    for i in range(1, n):
        elapsed = timestamp[i] - timestamp[last_good]
        if elapsed <= 1e-6:
            elapsed = dt[i - 1]
        rel = mink.SO3(out[last_good]).inverse().multiply(mink.SO3(out[i]))
        deg_per_s = np.degrees(np.linalg.norm(rel.log())) / elapsed
        if deg_per_s > MAX_ANGULAR_VELOCITY_DEG_S:
            bad[i] = True
        else:
            last_good = i

    if not bad.any():
        return out

    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        run_start = i
        while i < n and bad[i]:
            i += 1
        run_end = i  # first good frame after the run (exclusive), or n if none

        good_before = run_start - 1
        good_after = run_end if run_end < n else None

        if good_before < 0 and good_after is None:
            break  # whole stream flagged -- nothing sane to anchor to, leave as-is
        elif good_before < 0:
            out[run_start:run_end] = out[good_after]
        elif good_after is None:
            out[run_start:run_end] = out[good_before]
        else:
            q0, q1 = mink.SO3(out[good_before]), mink.SO3(out[good_after])
            t0, t1 = timestamp[good_before], timestamp[good_after]
            for k in range(run_start, run_end):
                frac = float(np.clip((timestamp[k] - t0) / (t1 - t0), 0.0, 1.0)) if t1 > t0 else 0.0
                out[k] = q0.interpolate(q1, frac).wxyz

    return out


def _corrected_target(correction: "mink.SE3", quat_wxyz: np.ndarray, pos_xyz: np.ndarray):
    T = correction.multiply(mink.SE3.from_rotation_and_translation(mink.SO3(quat_wxyz), pos_xyz))
    return T.rotation().wxyz, T.translation()


def align_action_to_obs(fk_ee: np.ndarray, action_ee: np.ndarray) -> np.ndarray:
    """Removes the per-episode, per-arm constant SE3 bias between a raw
    recorded action.{left,right}_ee stream and this MJCF's own site (the
    same phenomenon _anchor_correction exists for on the IK side, per §5's
    "Important caveat" -- e.g. observed directly: "pass the yellow cube"/
    "assemble the two gears" each show a consistent ~tens-of-cm per-arm
    Z offset between action.{left,right}_ee and the FK of that same
    episode's observation.state). Anchors frame 0's recorded target to frame
    0's FK'd pose -- ground truth, since observation.state is the real
    encoders -- and applies that same rigid transform to every subsequent
    frame, so the two streams become directly comparable (e.g. for the
    obs-vs-action-ee visualizer).

    fk_ee, action_ee: (N, 7) qw,qx,qy,qz,x,y,z -> (N, 7) corrected action_ee,
    same convention."""
    seed_T = mink.SE3.from_rotation_and_translation(mink.SO3(fk_ee[0, 0:4]), fk_ee[0, 4:7])
    correction = _anchor_correction(seed_T, action_ee[0, 0:4], action_ee[0, 4:7])

    out = np.zeros_like(action_ee)
    for i in range(action_ee.shape[0]):
        out[i, 0:4], out[i, 4:7] = _corrected_target(correction, action_ee[i, 0:4], action_ee[i, 4:7])
    return out


def compute_action_q(
    model: mujoco.MjModel,
    observation_state: np.ndarray,
    action_left_ee: np.ndarray,
    action_right_ee: np.ndarray,
    timestamp: np.ndarray | None = None,
) -> dict:
    """Runs the sequential, warm-started per-arm IK solve (§10) over every
    frame of one episode. observation_state: (N,15), action_*_ee: (N,7).
    timestamp: (N,) seconds, used only to rate-check _declitch_quaternions;
    defaults to a uniform 30fps grid if not given (every episode observed so
    far is uniformly sampled at that rate, so this is a safe fallback, not
    an approximation that changes behavior in practice)."""
    n = observation_state.shape[0]
    if timestamp is None:
        timestamp = np.arange(n, dtype=float) / 30.0
    lift_dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift_joint")]

    action_left_ee = action_left_ee.copy()
    action_right_ee = action_right_ee.copy()
    action_left_ee[:, 0:4] = _declitch_quaternions(action_left_ee[:, 0:4], timestamp)
    action_right_ee[:, 0:4] = _declitch_quaternions(action_right_ee[:, 0:4], timestamp)

    left_ik = ArmIK(model, "left", lift_dof)
    right_ik = ArmIK(model, "right", lift_dof)

    q0 = build_full_qpos(model, observation_state[0])
    left_ik.init(q0.copy())
    right_ik.init(q0.copy())

    left_seed_T = left_ik.configuration.get_transform_frame_to_world("left_arm_ee", "site")
    right_seed_T = right_ik.configuration.get_transform_frame_to_world("right_arm_ee", "site")
    left_correction = _anchor_correction(left_seed_T, action_left_ee[0, 0:4], action_left_ee[0, 4:7])
    right_correction = _anchor_correction(right_seed_T, action_right_ee[0, 0:4], action_right_ee[0, 4:7])

    qtarget_left = np.zeros((n, 7))
    qtarget_right = np.zeros((n, 7))
    left_solved = np.zeros(n, dtype=bool)
    right_solved = np.zeros(n, dtype=bool)
    left_pos_err = np.zeros(n)
    left_rot_err = np.zeros(n)
    right_pos_err = np.zeros(n)
    right_rot_err = np.zeros(n)

    left_unsolved_streak = 0
    right_unsolved_streak = 0

    for i in range(n):
        l_quat, l_pos = _corrected_target(left_correction, action_left_ee[i, 0:4], action_left_ee[i, 4:7])
        r_quat, r_pos = _corrected_target(right_correction, action_right_ee[i, 0:4], action_right_ee[i, 4:7])
        qtarget_left[i], left_solved[i], left_pos_err[i], left_rot_err[i] = left_ik.solve(l_quat, l_pos)
        qtarget_right[i], right_solved[i], right_pos_err[i], right_rot_err[i] = right_ik.solve(r_quat, r_pos)

        # See RESEED_AFTER_UNSOLVED: bound how long a warm-start stall (from
        # a glitch _declitch_quaternions missed, or a genuinely hard reach)
        # can drag on by snapping back to the real recorded pose.
        left_unsolved_streak = 0 if left_solved[i] else left_unsolved_streak + 1
        right_unsolved_streak = 0 if right_solved[i] else right_unsolved_streak + 1
        if left_unsolved_streak >= RESEED_AFTER_UNSOLVED:
            left_ik.reground(build_full_qpos(model, observation_state[i]))
            left_unsolved_streak = 0
        if right_unsolved_streak >= RESEED_AFTER_UNSOLVED:
            right_ik.reground(build_full_qpos(model, observation_state[i]))
            right_unsolved_streak = 0

    return {
        "qtarget_left": qtarget_left,
        "qtarget_right": qtarget_right,
        "left_solved": left_solved,
        "right_solved": right_solved,
        "left_pos_err": left_pos_err,
        "left_rot_err": left_rot_err,
        "right_pos_err": right_pos_err,
        "right_rot_err": right_rot_err,
    }
