"""
Automatic Face Matching Module

This module automatically matches a reference face to the best matching identity cluster
from detected faces, eliminating the need for manual ID selection.
"""

import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional


def find_matching_identity(
    reference_face_path: str,
    by_identity_dir: str,
    read_image_fn,
    get_faces_fn,
    compare_faces_fn,
    face_distance_threshold: float = 0.4,
    verbose: bool = True
) -> Optional[Tuple[str, str, float]]:
    """
    Find which identity cluster best matches the given reference face.

    Args:
        reference_face_path: Path to the reference face image
        by_identity_dir: Directory containing ID_xx folders
        read_image_fn: Function to read images (e.g., read_static_image)
        get_faces_fn: Function to detect faces (e.g., get_many_faces)
        compare_faces_fn: Function to compare faces (e.g., compare_faces)
        face_distance_threshold: Maximum distance to consider a match
        verbose: Whether to print progress information

    Returns:
        Tuple of (identity_name, sample_face_path, distance) or None if no match
    """
    if verbose:
        print(f"Finding identity match for: {os.path.basename(reference_face_path)}")
        print("="*60)

    # Load reference face
    ref_frame = read_image_fn(reference_face_path)
    if ref_frame is None:
        if verbose:
            print(f"✗ Could not read reference image")
        return None

    ref_faces = get_faces_fn([ref_frame])
    if not ref_faces:
        if verbose:
            print(f"✗ No face detected in reference image")
        return None

    reference_face = ref_faces[0]

    # Get all identity folders
    identity_folders = sorted(glob.glob(os.path.join(by_identity_dir, "ID_*")))

    if not identity_folders:
        if verbose:
            print(f"✗ No identity folders found in {by_identity_dir}")
        return None

    best_match = None
    best_distance = float('inf')

    # Check each identity cluster
    for identity_folder in identity_folders:
        identity_name = os.path.basename(identity_folder)

        # Get sample images from this identity (check first few)
        image_paths = sorted(
            glob.glob(os.path.join(identity_folder, "*.png")) +
            glob.glob(os.path.join(identity_folder, "*.jpg"))
        )[:5]  # Only check first 5 images for speed

        if not image_paths:
            continue

        # Try to match against samples from this identity
        for img_path in image_paths:
            frame = read_image_fn(img_path)
            if frame is None:
                continue

            faces = get_faces_fn([frame])
            if not faces:
                continue

            cluster_face = faces[0]

            # Compare faces using embedding distance
            if compare_faces_fn(face=reference_face, reference_face=cluster_face, face_distance=face_distance_threshold):
                # Calculate actual distance for ranking
                distance = np.linalg.norm(
                    np.array(reference_face.embedding) - np.array(cluster_face.embedding)
                )

                if distance < best_distance:
                    best_distance = distance
                    best_match = (identity_name, img_path, distance)

                if verbose:
                    print(f"  ✓ {identity_name}: Match found (distance={distance:.3f})")
                break  # Found match in this cluster, move to next
        else:
            if verbose:
                print(f"  ✗ {identity_name}: No match")

    if verbose:
        print("="*60)
        if best_match:
            print(f"✓ Best match: {best_match[0]} (distance={best_match[2]:.3f})")
        else:
            print(f"✗ No matching identity found")
        print()

    return best_match


def auto_map_faces_to_identities(
    face_mappings: Dict[str, str],
    by_identity_dir: str,
    read_image_fn,
    get_faces_fn,
    compare_faces_fn,
    face_distance_threshold: float = 0.4,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Automatically map multiple reference faces to their matching identity clusters.

    Args:
        face_mappings: Dict mapping label -> reference_face_path (e.g., {"boy1": "path/to/boy1.jpg"})
        by_identity_dir: Directory containing ID_xx folders
        read_image_fn: Function to read images
        get_faces_fn: Function to detect faces
        compare_faces_fn: Function to compare faces
        face_distance_threshold: Maximum distance to consider a match
        verbose: Whether to print progress

    Returns:
        Dict mapping label -> identity_name (e.g., {"boy1": "ID_18"})
    """
    identity_map = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Auto-mapping {len(face_mappings)} reference faces to identities")
        print(f"{'='*60}\n")

    for label, ref_face_path in face_mappings.items():
        match = find_matching_identity(
            reference_face_path=ref_face_path,
            by_identity_dir=by_identity_dir,
            read_image_fn=read_image_fn,
            get_faces_fn=get_faces_fn,
            compare_faces_fn=compare_faces_fn,
            face_distance_threshold=face_distance_threshold,
            verbose=verbose
        )

        if match:
            identity_name, sample_path, distance = match
            identity_map[label] = identity_name
        else:
            if verbose:
                print(f"⚠️  Warning: No match found for {label}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Mapping Summary:")
        print(f"{'='*60}")
        for label, identity in identity_map.items():
            print(f"  {label} → {identity}")
        print(f"{'='*60}\n")

    return identity_map


def get_identity_face_paths(
    identity_map: Dict[str, str],
    by_identity_dir: str
) -> Dict[str, List[str]]:
    """
    Get all face paths for each mapped identity.

    Args:
        identity_map: Dict from auto_map_faces_to_identities
        by_identity_dir: Directory containing ID_xx folders

    Returns:
        Dict mapping label -> list of face image paths
    """
    face_paths = {}

    for label, identity_name in identity_map.items():
        identity_folder = os.path.join(by_identity_dir, identity_name)

        if os.path.exists(identity_folder):
            paths = sorted(
                glob.glob(os.path.join(identity_folder, "*.png")) +
                glob.glob(os.path.join(identity_folder, "*.jpg"))
            )
            face_paths[label] = paths
        else:
            face_paths[label] = []

    return face_paths


def create_swap_config(
    identity_map: Dict[str, str],
    reference_faces: Dict[str, str],
    by_identity_dir: str,
    use_best_reference: bool = True,
    best_refs_dir: Optional[str] = None
) -> Dict[str, Dict[str, any]]:
    """
    Create a complete face swap configuration.

    Args:
        identity_map: Dict from auto_map_faces_to_identities (label -> identity)
        reference_faces: Dict of label -> new_face_path to swap to
        by_identity_dir: Directory containing ID_xx folders
        use_best_reference: If True, use best quality face from cluster
        best_refs_dir: Directory containing best reference faces (if available)

    Returns:
        Dict with swap configuration for each label
    """
    swap_config = {}

    for label, identity_name in identity_map.items():
        identity_folder = os.path.join(by_identity_dir, identity_name)

        # Get original faces to swap
        original_faces = sorted(
            glob.glob(os.path.join(identity_folder, "*.png")) +
            glob.glob(os.path.join(identity_folder, "*.jpg"))
        )

        # Get the face to use as reference (for matching)
        if use_best_reference and best_refs_dir:
            # Use the best quality face
            best_ref = os.path.join(best_refs_dir, f"{identity_name}_best.png")
            if not os.path.exists(best_ref):
                best_ref = original_faces[0] if original_faces else None
            reference_face = best_ref
        else:
            # Use first face from cluster
            reference_face = original_faces[0] if original_faces else None

        swap_config[label] = {
            "identity": identity_name,
            "new_face": reference_faces[label],
            "original_faces": original_faces,
            "reference_face": reference_face,
            "num_faces": len(original_faces)
        }

    return swap_config
