import os 
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from collections import defaultdict
from skin_segmentation import skin_mask_ycrcb, skin_ratio_in_bbox


class Detection:
    """ Create a class for information about the detected image.
      Track the box around the image, location of the face points 
      and warped face. Embeddings are used for the bulk processing. """
    def __init__(self, bbox, landmarks, aligned_crop, embedding=None):
        self.bbox = bbox
        self.landmarks = landmarks 
        self.aligned_crop = aligned_crop
        self.embedding = embedding

class FaceProcessor:

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=5)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(160, 160))

    def embed(self, crop_bgr: np.ndarray):
        """Return an L2-normalised 512-D vector, or None if no face is detected."""
        upscaled = cv2.resize(crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
        faces = self.app.get(upscaled)
        if not faces:
            return None
        return faces[0].normed_embedding
    
    def process_faces(self, folder_name: str, total_faces: int = 1):
        """
        faces_folder: list containing all image files
        total_faces:  the # of faces across all files
        """
        
        # build an output folder - ignore if already exists 
        os.makedirs("output_faces", exist_ok=True) 

        # collect image paths
        faces_folder = list(Path(folder_name).glob("*"))

        cols = 3 if total_faces > 2 else 1
        rows = int(np.ceil(total_faces / cols))

        face_idx = 0

        # iterate through each image in the folder 
        for idx in range(len(faces_folder)):

            print(f"index: {idx}\n")
            img = cv2.imread(faces_folder[idx])
            img_w, img_h = img.shape[1], img.shape[0]

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            )

            detection_result = self.detector.detect(mp_image)

            if len(detection_result.face_landmarks) == 0: 
                print(f"[ERROR] Could not find a face in {faces_folder[idx]}")
                continue

            face_count = len(detection_result.face_landmarks)
            print(f"{faces_folder[idx]}: {face_count}")

            # iterate through each face in each image 
            for face in detection_result.face_landmarks:
                
                # Extract known features
                left_eye_center = face[468]
                right_eye_center = face[473]
                nose_tip = face[4]

                # De-normalize coordinates to allow indexing original image 
                left_eye_x = int(left_eye_center.x * img_w)
                left_eye_y = int(left_eye_center.y * img_h)

                right_eye_x = int(right_eye_center.x * img_w)
                right_eye_y = int(right_eye_center.y * img_h)

                nose_tip_x = int(nose_tip.x * img_w)
                nose_tip_y = int(nose_tip.y * img_h)

                # Feature coordinates in original image 
                left_eye = (left_eye_x , left_eye_y)
                right_eye = (right_eye_x, right_eye_y)
                nose = (nose_tip_x, nose_tip_y)

                # Target coordinates for feature transform (from specifications doc)
                new_left_eye = (40, 40)
                new_right_eye = (85, 40)
                new_nose = (63, 70)

                # Source array
                src = np.array([
                    left_eye,
                    right_eye,
                    nose
                ], dtype=np.float32)

                # Destination array
                dst = np.array([
                    new_left_eye, 
                    new_right_eye,
                    new_nose
                ], dtype=np.float32)
                
                # Estimate coordinate tranformation required to take source points to destination   
                M, _ = cv2.estimateAffinePartial2D(src, dst)

                # Compute affine transformation on original image - focused around a 125x125 window
                aligned_img = cv2.warpAffine(img, M, (125, 125))

                # Dynamic filename to handle multiple faces per image 
                filename = f"img_{idx}_face_{face_idx}.jpg"
                output_path = f"output_faces/{filename}" 
                cv2.imwrite(output_path, aligned_img) # store output image 
                                
                face_idx += 1 # onto the next face

    def similarity(self, crops_folder: str = "output_faces", eps: float = 0.7):
        """
        Embed every crop in `crops_folder`, cluster with DBSCAN on cosine distance,
        and return groupings.

        Returns:
            groups: dict mapping cluster_id -> list of filenames
            labels: list of (filename, cluster_id) in input order
        """
        crop_paths = sorted(Path(crops_folder).glob("*.jpg"))

        embeddings, names = [], []
        for p in crop_paths:
            crop = cv2.imread(str(p))
            if crop is None:
                continue
            emb = self.embed(crop)
            if emb is None:
                print(f"[skip] no embedding for {p.name}")
                continue
            embeddings.append(emb)
            names.append(p.name)

        if not embeddings:
            return {}, []

        X = np.vstack(embeddings)
        cluster_ids = DBSCAN(eps=eps, metric="cosine", min_samples=1).fit(X).labels_

        groups = defaultdict(list)
        labels = []
        for name, cid in zip(names, cluster_ids):
            groups[int(cid)].append(name)
            labels.append((name, int(cid)))

        print(f"\n{len(names)} faces → {len(groups)} identities\n")
        for cid in sorted(groups):
            print(f"  identity {cid}:")
            for n in groups[cid]:
                print(f"    {n}")

        return dict(groups), labels
    
    def detect_one(self, img_bgr: np.ndarray, do_embed: bool = True,
                   skin_threshold: float = 0.1) -> list:
        """
        Run MediaPipe on a single BGR image and return one Detection per face. 
        """
        img_h, img_w = img_bgr.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        )
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return[]
        detections = []
        for face in result.face_landmarks:
            # Indicies 468/473 are iris centres 
            # MediaPipe returns normalised [0-1] coords so can scale to pixels
            left_eye = (int(face[468].x * img_w), int(face[468].y * img_h))
            right_eye = (int(face[473].x * img_w), int(face[473].y * img_h))
            nose = (int(face[4].x * img_w), int(face[4].y * img_h))

            # Bounding box fr min max of all landmark pixels 
            xs = [int(lm.x * img_w) for lm in face]
            ys = [int(lm.y * img_h) for lm in face]
            bbox = (max(0, min(xs)), max(0, min(ys)),
                    min(img_w - 1, max(xs)), min(img_h -1, max(ys)))
            
            # Filter out detections with very low skin pixel ratio (false positives)
            # Low threshold to about rejecting darker skin tones
            skin_mask = skin_mask_ycrcb(img_bgr)
            if skin_ratio_in_bbox(skin_mask, bbox) < skin_threshold:
                continue
            
            # Map the detected eye and nose positions onto fixed canonical positions
            src = np.array([left_eye, right_eye, nose], dtype=np.float32)
            dst = np.array([(40, 40), (85, 40), (63, 70)], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(src, dst)
            aligned_crop = cv2.warpAffine(img_bgr, M, (125, 125))

            # Compute embedding for bulk mode 
            embedding = self.embed(aligned_crop) if do_embed else None

            detections.append(Detection(
                bbox=bbox,
                landmarks={"left_eye": left_eye, "right_eye": right_eye, "nose": nose},
                aligned_crop=aligned_crop,
                embedding=embedding,
            ))
        return detections
    
    def similarity_from_embeddings(self, embeddings: list, eps: float = 0.6) -> list:
        """Cluster embedding vectors and return one integer label per embedding."""
        X = np.vstack(embeddings)
        # min_samples=1 means no face is ever an outlier so always joins cluster
        # eps controls similarity threshold
        labels = DBSCAN(eps=eps, metric="cosine", min_samples=1).fit(X).labels_
        return [int(l) for l in labels]
    
    def plot_clusters_inmem(self, crops: list, labels: list, block: bool = True) -> None:
        """
        Plot one row per identity cluster using crop arrays directly.
        """
        # Group the index of each crop by its cluster label
        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[label].append(i)
        
        n_rows = len(groups)
        n_cols = max(len(v) for v in groups.values())
        _, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.2 * n_rows), squeeze=False)

        for r, cid in enumerate(sorted(groups)):
            for c in range(n_cols):
                axes[r, c].axis("off")
                if c < len(groups[cid]):
                    # Convert BGR to RGB for matplotlib display 
                    axes[r, c].imshow(cv2.cvtColor(crops[groups[cid][c]], cv2.COLOR_BGR2RGB))
            axes[r, 0].set_ylabel(f"identity {cid}", rotation=0, ha="right", va="center", fontsize=9)
        
        plt.tight_layout()
        plt.show(block=block)


    def plot_clusters(self, groups: dict, crops_folder: str = "output_faces"):
        """Plot one row per identity cluster, showing each face in that cluster."""
        if not groups:
            print("Nothing to plot.")
            return

        n_rows = len(groups)
        n_cols = max(len(v) for v in groups.values())

        _, axes = plt.subplots(
            n_rows, n_cols, figsize=(2 * n_cols, 2.2 * n_rows), squeeze=False
        )
        for r, cid in enumerate(sorted(groups)):
            for c in range(n_cols):
                axes[r, c].axis("off")
                if c < len(groups[cid]):
                    name = groups[cid][c]
                    crop = cv2.imread(str(Path(crops_folder) / name))
                    axes[r, c].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    axes[r, c].set_title(name, fontsize=7)
            axes[r, 0].set_ylabel(
                f"identity {cid}", rotation=0, ha="right", va="center", fontsize=9
            )

        plt.tight_layout()
        plt.show()

