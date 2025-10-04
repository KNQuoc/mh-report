import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Literal
import json
import cv2
import math
import os
from dataclasses import dataclass

# Data classes for results
@dataclass
class SwapSuccess:
    outputs: list[str]
    success: Literal[True] = True

@dataclass
class SwapFailure:
    error: str
    success: Literal[False] = False

def compute_embedding_distance(emb1, emb2):
    """Compute cosine distance between two embeddings"""
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

class ImprovedIdentityTracker:
    def __init__(self, distance_threshold=0.4):
        self.identities = {}  # identity_id -> list of embeddings
        self.identity_centroids = {}  # identity_id -> centroid embedding
        self.next_id = 0
        self.distance_threshold = distance_threshold

    def add_face(self, face_embedding, face_data):
        """Add a face and return its identity ID"""
        if not self.identities:
            # First face
            self.identities[0] = [face_embedding]
            self.identity_centroids[0] = face_embedding
            self.next_id = 1
            return 0

        # Find closest identity using centroids
        min_dist = float('inf')
        best_id = None

        for id, centroid in self.identity_centroids.items():
            dist = compute_embedding_distance(face_embedding, centroid)
            if dist < min_dist:
                min_dist = dist
                best_id = id

        if min_dist < self.distance_threshold:
            # Add to existing identity and update centroid
            self.identities[best_id].append(face_embedding)
            # Update centroid as average of all embeddings
            self.identity_centroids[best_id] = np.mean(self.identities[best_id], axis=0)
            return best_id
        else:
            # Create new identity
            new_id = self.next_id
            self.identities[new_id] = [face_embedding]
            self.identity_centroids[new_id] = face_embedding
            self.next_id += 1
            return new_id

    def cluster_with_dbscan(self, all_embeddings, eps=0.4, min_samples=2):
        """Use DBSCAN to cluster all faces"""
        if len(all_embeddings) < 2:
            return [0] * len(all_embeddings)

        # Compute distance matrix
        n = len(all_embeddings)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = compute_embedding_distance(all_embeddings[i], all_embeddings[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distances)

        # Handle noise points (-1 labels) by assigning them to nearest cluster
        for i, label in enumerate(labels):
            if label == -1:
                # Find nearest non-noise point
                min_dist = float('inf')
                best_label = 0
                for j, other_label in enumerate(labels):
                    if other_label != -1 and distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        best_label = other_label
                labels[i] = best_label

        return labels

# Basic face quality scoring function
def face_quality_score(frame: np.ndarray, face) -> tuple[float, dict]:
    """
    Composite score of size, sharpness, frontalness.
    """
    # compute a padded bbox for the area metric directly from the face bbox
    sx, sy, ex, ey = map(int, face.bounding_box)  # type: ignore
    pad_ratio = 0.25
    px, py = int((ex - sx) * pad_ratio), int((ey - sy) * pad_ratio)
    sx, sy = max(0, sx - px), max(0, sy - py)
    ex, ey = max(0, ex + px), max(0, ey + py)

    H, W = frame.shape[:2]
    area_norm = ((ex - sx) * (ey - sy)) / max(1.0, float(W * H))  # larger face => higher

    # robustly get the cropped patch
    crop = frame[sy:ey, sx:ex]

    # sharpness via Laplacian variance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_norm = min(1.0, lap / 300.0)  # tune 300 for your footage

    # frontalness from landmarks if present
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
            mid_x = 0.5 * (L[0] + R[0]); half_eye = 0.5 * abs(R[0] - L[0]) + 1e-6
            yaw_norm = abs(nose[0] - mid_x) / half_eye
            yaw_score = max(0.0, 1.0 - min(1.0, yaw_norm))
            frontal = 0.5 * roll_score + 0.5 * yaw_score
        except Exception:
            frontal = 0.5
    else:
        frontal = 0.5

    score = 0.4 * area_norm + 0.4 * sharp_norm + 0.2 * frontal
    return score, {"area": area_norm, "sharp": sharp_norm, "frontal": frontal}

# Extract faces from image helper function
def extract_faces_from_image(
    vision_frame,
    face,
    output_dir: str,
    frame_number: int,
    face_idx: int,
) -> str:
    """Extract and save a face region from an image frame."""
    from PIL import Image
    from facefusion.vision import normalize_frame_color

    start_x, start_y, end_x, end_y = map(int, face.bounding_box)  # type: ignore
    padding_x = int((end_x - start_x) * 0.25)
    padding_y = int((end_y - start_y) * 0.25)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    crop_vision_frame = vision_frame[start_y:end_y, start_x:end_x]
    crop_vision_frame = normalize_frame_color(crop_vision_frame)

    face_filename = os.path.join(output_dir, f"{frame_number}-{face_idx}.png")
    img = Image.fromarray(crop_vision_frame)
    img.save(face_filename)

    return face_filename

# Enhanced face quality scoring
def enhanced_face_quality_score(frame, face, embeddings_history=None):
    """
    Enhanced quality score considering:
    - Size, sharpness, frontalness (existing)
    - Embedding consistency (if history available)
    """
    # Get basic quality score
    basic_score, components = face_quality_score(frame, face)

    if embeddings_history and len(embeddings_history) > 0:
        # Calculate embedding consistency
        face_emb = face.embedding if hasattr(face, 'embedding') else face.normed_embedding

        # Average distance to previous embeddings of same identity
        distances = [compute_embedding_distance(face_emb, prev_emb)
                    for prev_emb in embeddings_history[-5:]]  # Last 5 embeddings
        consistency_score = max(0, 1 - np.mean(distances))

        # Weighted combination
        final_score = 0.7 * basic_score + 0.3 * consistency_score
        components['consistency'] = consistency_score
    else:
        final_score = basic_score

    return final_score, components

# Integration with your existing process_video function
def process_video_with_improved_clustering(
    task_id: str,
    file_path: str,
    output_dir: str,
    handle_complete,
    face_detector_score: float,
    use_dbscan: bool = True
):
    """Enhanced version with better identity tracking"""
    import logging, time, cv2, os
    from facefusion.vision import read_video_frame
    from facefusion.face_analyser import get_many_faces
    from facefusion import state_manager

    # Initialize state manager with required parameters
    state_manager.init_item("face_detector_score", face_detector_score)
    state_manager.init_item("target_path", file_path)

    def log(msg):
        logging.info(f"[{task_id}] {msg}")

    # Initialize tracker
    tracker = ImprovedIdentityTracker(distance_threshold=0.35)

    # Prepare directories
    all_dir = os.path.join(output_dir, "all")
    byid_dir = os.path.join(output_dir, "by_identity")
    final_dir = os.path.join(output_dir, "final")

    for d in [all_dir, byid_dir, final_dir]:
        os.makedirs(d, exist_ok=True)

    # Collect all faces and embeddings first
    all_faces = []
    all_embeddings = []
    all_frames = []
    all_frame_indices = []

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_skip = max(1, int(fps / 2))  # Sample at 2 FPS

    log(f"Collecting faces from {total_frames} frames")

    current_frame = 0
    while current_frame < total_frames:
        frame = read_video_frame(file_path, current_frame)
        if frame is None:
            current_frame += frames_to_skip
            continue

        faces = get_many_faces([frame])

        for idx, face in enumerate(faces):
            # Get embedding
            if hasattr(face, 'embedding'):
                emb = face.embedding
            elif hasattr(face, 'normed_embedding'):
                emb = face.normed_embedding
            else:
                continue

            all_faces.append(face)
            all_embeddings.append(emb)
            all_frames.append(frame)
            all_frame_indices.append((current_frame, idx))

        current_frame += frames_to_skip

    cap.release()

    log(f"Collected {len(all_faces)} total face detections")

    # Cluster using DBSCAN or multi-embedding tracker
    if use_dbscan and len(all_embeddings) > 1:
        log("Clustering with DBSCAN")
        labels = tracker.cluster_with_dbscan(all_embeddings, eps=0.35, min_samples=2)
    else:
        log("Using multi-embedding tracker")
        labels = []
        for emb, face in zip(all_embeddings, all_faces):
            label = tracker.add_face(emb, face)
            labels.append(label)

    # Group by identity and find best exemplar
    identities = {}
    for i, label in enumerate(labels):
        if label not in identities:
            identities[label] = []
        identities[label].append(i)

    log(f"Found {len(identities)} unique identities")

    # Save best face for each identity
    final_paths = []
    for identity_id, indices in identities.items():
        # Find best quality face for this identity
        best_score = -1
        best_idx = indices[0]

        # Get embedding history for this identity
        identity_embeddings = [all_embeddings[i] for i in indices]

        for idx in indices:
            face = all_faces[idx]
            frame = all_frames[idx]

            score, _ = enhanced_face_quality_score(frame, face, identity_embeddings)

            if score > best_score:
                best_score = score
                best_idx = idx

        # Save best face
        face = all_faces[best_idx]
        frame = all_frames[best_idx]
        frame_num, face_idx = all_frame_indices[best_idx]

        output_path = extract_faces_from_image(
            vision_frame=frame,
            face=face,
            output_dir=final_dir,
            frame_number=frame_num,
            face_idx=identity_id
        )
        final_paths.append(output_path)
        handle_complete(output_path)

        # Save all faces for this identity
        id_dir = os.path.join(byid_dir, f"ID_{identity_id:02d}")
        os.makedirs(id_dir, exist_ok=True)

        for idx in indices:
            face = all_faces[idx]
            frame = all_frames[idx]
            frame_num, face_idx = all_frame_indices[idx]

            extract_faces_from_image(
                vision_frame=frame,
                face=face,
                output_dir=id_dir,
                frame_number=frame_num,
                face_idx=face_idx
            )

    # Save clustering metadata
    metadata = {
        "total_detections": len(all_faces),
        "unique_identities": len(identities),
        "clustering_method": "DBSCAN" if use_dbscan else "multi-embedding",
        "identity_sizes": {f"ID_{k:02d}": len(v) for k, v in identities.items()}
    }

    with open(os.path.join(output_dir, "clustering_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"Processing complete: {len(identities)} identities from {len(all_faces)} detections")

    return SwapSuccess(outputs=final_paths)