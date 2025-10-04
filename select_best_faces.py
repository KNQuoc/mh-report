"""
Best Face Selection Module

This module selects the highest quality face from each identity cluster
based on size, sharpness, and frontalness metrics.
"""

import os
import glob
import shutil
import numpy as np
import cv2
import math
from typing import List, Dict, Tuple, Optional


def face_quality_score(frame: np.ndarray, face) -> Tuple[float, dict]:
    """
    Composite score of size, sharpness, frontalness.

    Args:
        frame: The image frame containing the face
        face: Face object with bounding_box and landmarks

    Returns:
        Tuple of (overall_score, metrics_dict)
    """
    # Compute a padded bbox for the area metric directly from the face bbox
    sx, sy, ex, ey = map(int, face.bounding_box)
    pad_ratio = 0.25
    px, py = int((ex - sx) * pad_ratio), int((ey - sy) * pad_ratio)
    sx, sy = max(0, sx - px), max(0, sy - py)
    ex, ey = max(0, ex + px), max(0, ey + py)

    H, W = frame.shape[:2]
    area_norm = ((ex - sx) * (ey - sy)) / max(1.0, float(W * H))  # larger face => higher

    # Robustly get the cropped patch
    crop = frame[sy:ey, sx:ex]

    # Sharpness via Laplacian variance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_norm = min(1.0, lap / 300.0)  # tune 300 for your footage

    # Frontalness from landmarks if present
    landmarks = getattr(face, "landmarks", None) or getattr(face, "landmark", None)
    if landmarks is not None:
        lm = np.array(landmarks, dtype=np.float32).reshape(-1, 2)
        try:
            if lm.shape[0] >= 48:
                L = lm[[36,37,38,39,40,41]].mean(axis=0)
                R = lm[[42,43,44,45,46,47]].mean(axis=0)
                nose = lm[30] if lm.shape[0] > 30 else lm[lm.shape[0]//2]
            else:
                L = lm[:lm.shape[0]//2].mean(axis=0)
                R = lm[lm.shape[0]//2:].mean(axis=0)
                nose = lm[lm.shape[0]//2]
            dx, dy = (R - L)
            roll_deg = abs(math.degrees(math.atan2(dy, dx)))
            roll_score = max(0.0, 1.0 - (roll_deg / 30.0))
            mid_x = 0.5 * (L[0] + R[0])
            half_eye = 0.5 * abs(R[0] - L[0]) + 1e-6
            yaw_norm = abs(nose[0] - mid_x) / half_eye
            yaw_score = max(0.0, 1.0 - min(1.0, yaw_norm))
            frontal = 0.5 * roll_score + 0.5 * yaw_score
        except Exception:
            frontal = 0.5
    else:
        frontal = 0.5

    score = 0.4 * area_norm + 0.4 * sharp_norm + 0.2 * frontal
    return score, {"area": area_norm, "sharp": sharp_norm, "frontal": frontal}


def select_best_face_per_identity(
    by_identity_dir: str,
    read_image_fn,
    get_faces_fn,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Select the best quality face image from each identity cluster.

    Args:
        by_identity_dir: Path to directory containing ID_xx folders
        read_image_fn: Function to read images (e.g., facefusion.vision.read_static_image)
        get_faces_fn: Function to detect faces (e.g., facefusion.face_analyser.get_many_faces)
        verbose: Whether to print progress information

    Returns:
        Dictionary mapping identity_name -> path_to_best_face_image
    """
    best_faces = {}

    # Iterate through each identity folder (ID_00, ID_01, etc.)
    identity_folders = sorted(glob.glob(os.path.join(by_identity_dir, "ID_*")))

    if verbose:
        print(f"Found {len(identity_folders)} identity clusters")
        print("="*60)

    for identity_folder in identity_folders:
        identity_name = os.path.basename(identity_folder)
        if verbose:
            print(f"\nProcessing {identity_name}...")

        # Get all images in this identity folder
        image_paths = sorted(
            glob.glob(os.path.join(identity_folder, "*.png")) +
            glob.glob(os.path.join(identity_folder, "*.jpg"))
        )

        if not image_paths:
            if verbose:
                print(f"  ⚠️  No images found in {identity_name}")
            continue

        if verbose:
            print(f"  Evaluating {len(image_paths)} faces...")

        best_score = -1
        best_path = None
        best_metrics = None

        # Evaluate each face in the cluster
        for img_path in image_paths:
            frame = read_image_fn(img_path)
            if frame is None:
                continue

            faces = get_faces_fn([frame])
            if not faces:
                continue

            # Use the first/main face in the image
            face = faces[0]

            # Calculate quality score
            try:
                score, metrics = face_quality_score(frame, face)

                if score > best_score:
                    best_score = score
                    best_path = img_path
                    best_metrics = metrics
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Error scoring {os.path.basename(img_path)}: {e}")
                continue

        if best_path:
            best_faces[identity_name] = best_path
            if verbose:
                print(f"  ✓ Best face: {os.path.basename(best_path)}")
                print(f"    Score: {best_score:.3f} (area={best_metrics['area']:.3f}, sharp={best_metrics['sharp']:.3f}, frontal={best_metrics['frontal']:.3f})")
        else:
            if verbose:
                print(f"  ✗ No valid faces found in {identity_name}")

    if verbose:
        print("\n" + "="*60)
        print(f"✓ Selected best faces for {len(best_faces)}/{len(identity_folders)} identities")
        print("="*60)

    return best_faces


def copy_best_faces_to_output(
    best_faces: Dict[str, str],
    output_dir: str,
    folder_name: str = "best_references",
    verbose: bool = True
) -> str:
    """
    Copy the best face from each identity to a specified output folder.

    Args:
        best_faces: Dictionary from select_best_face_per_identity
        output_dir: Base output directory
        folder_name: Name of folder to create for best references
        verbose: Whether to print progress information

    Returns:
        Path to the folder containing best reference faces
    """
    best_refs_dir = os.path.join(output_dir, folder_name)
    os.makedirs(best_refs_dir, exist_ok=True)

    if verbose:
        print(f"\nCopying best faces to: {best_refs_dir}")
        print("="*60)

    for identity_name, img_path in sorted(best_faces.items()):
        # Copy with identity name preserved
        dest_path = os.path.join(best_refs_dir, f"{identity_name}_best.png")
        shutil.copy2(img_path, dest_path)
        if verbose:
            print(f"✓ {identity_name}: {os.path.basename(img_path)}")

    if verbose:
        print("="*60)
        print(f"✓ Copied {len(best_faces)} best reference faces")

    return best_refs_dir


def get_best_reference_paths(best_refs_dir: str, num_references: Optional[int] = None) -> List[str]:
    """
    Get paths to best reference faces, optionally limited to top N.

    Args:
        best_refs_dir: Directory containing best reference faces
        num_references: Maximum number of references to return (None = all)

    Returns:
        List of file paths to best reference faces
    """
    reference_paths = sorted(
        glob.glob(os.path.join(best_refs_dir, "ID_*_best.png")) +
        glob.glob(os.path.join(best_refs_dir, "ID_*_best.jpg"))
    )

    if num_references is not None:
        reference_paths = reference_paths[:num_references]

    return reference_paths
