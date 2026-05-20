import os 
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 


class FeatureExtractor:

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=5)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_faces(self, folder_name: str, total_faces: int):
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


