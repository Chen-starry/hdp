"""Project end-effector trajectories onto a camera image.

This script loads RLBench style episodes and visualises the gripper motion
by projecting the 3D end-effector positions into a 2D camera plane.  The
resulting trajectories are plotted over the RGB observation of the selected
camera, creating figures similar to those shown in the Diffusion Policy
paper.

Example
-------
python project_trajectories.py \
    --data-root /path/to/episodes \
    --camera front \
    --max-episodes 20 \
    --output diff_style_traj.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def _load_background_image(episode_dir: Path, camera: str) -> np.ndarray:
    """Load the first RGB frame for the requested camera."""

    img_path = episode_dir / f"{camera}_rgb" / "0.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Could not find background image: {img_path}")

    # cv2 loads images in BGR, convert them to RGB for matplotlib usage.
    return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)


def _project_world_points(
    world_points: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D world points into 2D pixel coordinates.

    RLBench stores camera extrinsics as a transformation from the camera frame
    to the world frame (``T_wc``).  To project world points we require the
    inverse transform (``T_cw``).  The conversion is performed explicitly to
    avoid numerical drift and to make the math clear.

    Parameters
    ----------
    world_points
        Array with shape ``(N, 3)`` containing points expressed in the world
        coordinate system.
    extrinsics
        ``4x4`` transformation matrix describing the pose of the camera in the
        world frame.
    intrinsics
        ``3x3`` camera intrinsics matrix.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple ``(pixel_points, camera_points)`` where ``pixel_points`` has
        shape ``(N, 2)`` and stores the projected pixel locations and
        ``camera_points`` has shape ``(N, 3)`` representing the same points in
        the camera coordinate system (useful for filtering points behind the
        camera).
    """

    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError("world_points must be of shape (N, 3)")

    if extrinsics.shape != (4, 4):
        raise ValueError("extrinsics must be a 4x4 matrix")

    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must be a 3x3 matrix")

    rotation_wc = extrinsics[:3, :3]
    translation_wc = extrinsics[:3, 3]

    # Transform points into the camera frame: X_c = R_cw * X_w + t_cw
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_cw @ translation_wc
    camera_points = (rotation_cw @ world_points.T).T + translation_cw

    # Use OpenCV to project onto the image plane.
    rvec, _ = cv2.Rodrigues(rotation_cw)
    projected, _ = cv2.projectPoints(
        world_points.astype(np.float32),
        rvec,
        translation_cw.astype(np.float32),
        intrinsics.astype(np.float32),
        distCoeffs=None,
    )

    pixel_points = projected.reshape(-1, 2)
    return pixel_points, camera_points


def _iter_episode_dirs(data_root: Path) -> Iterable[Path]:
    """Yield episode directories sorted by name."""

    for entry in sorted(data_root.iterdir()):
        if entry.is_dir():
            yield entry
def _load_camera_matrices(observation, camera: str) -> Tuple[np.ndarray, np.ndarray]:
    extrinsics = observation.misc[f"{camera}_camera_extrinsics"]
    intrinsics = observation.misc[f"{camera}_camera_intrinsics"]
    return extrinsics, intrinsics


def plot_trajectories(
    data_root: Path,
    camera: str,
    max_episodes: int,
    output_path: Path | None,
    show: bool,
) -> None:
    episode_dirs = list(_iter_episode_dirs(data_root))
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found under {data_root}")

    reference_episode = episode_dirs[0]
    background = _load_background_image(reference_episode, camera)
    height, width = background.shape[:2]

    fig, ax = plt.subplots(figsize=(width / 128, height / 128), dpi=128)
    ax.imshow(background)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip Y to match image coordinates.
    ax.axis("off")

    num_plotted = 0
    for episode_dir in episode_dirs:
        if num_plotted >= max_episodes:
            break

        pkl_path = episode_dir / "low_dim_obs.pkl"
        if not pkl_path.exists():
            continue

        with pkl_path.open("rb") as fh:
            observations = pickle.load(fh)

        if not observations:
            continue

        extrinsics, intrinsics = _load_camera_matrices(observations[0], camera)
        positions = np.asarray([obs.gripper_pose[:3] for obs in observations], dtype=np.float32)

        pixel_points, camera_points = _project_world_points(
            positions,
            extrinsics,
            intrinsics,
        )

        # Only draw points that are in front of the camera and inside the image.
        depth = camera_points[:, 2]
        valid = depth > 1e-6
        if not np.any(valid):
            continue

        pixel_points = pixel_points[valid]
        color_values = cm.plasma(np.linspace(0, 1, len(pixel_points)))

        # Clip to image boundaries for clean visualisations.
        pixel_points[:, 0] = np.clip(pixel_points[:, 0], 0, width - 1)
        pixel_points[:, 1] = np.clip(pixel_points[:, 1], 0, height - 1)

        for idx in range(1, len(pixel_points)):
            ax.plot(
                pixel_points[idx - 1 : idx + 1, 0],
                pixel_points[idx - 1 : idx + 1, 1],
                color=color_values[idx],
                linewidth=2,
                alpha=0.8,
            )

        num_plotted += 1

    fig.tight_layout(pad=0)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 3D end-effector trajectories into a camera plane.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path containing RLBench style episode folders.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        help="Camera name to use for background image and projection.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=20,
        help="Maximum number of episodes to render.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the generated figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure using matplotlib's interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_trajectories(
        data_root=args.data_root,
        camera=args.camera,
        max_episodes=args.max_episodes,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()

