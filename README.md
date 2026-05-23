# CITS4402 Face Detection & Matching

### Project Overview
Detects faces in images using MediaPipe landmarks, aligns them to a canonical 125x125 crop, embeds them with InsightFace and clusters them by identity using DBSCAN. There are two entry to this, the command line and the tkinter GUI.

### 1. Prerequisites
- Python 3.10+
- A `faces/` folder of `.jpg`/`.png` images to test with.

### 2.Create the virtual environment 

`python -m venv venv`
For mac/linux:
`source venv/bin/activate`
For Windows:
`venv\Scripts\activate` 

### 3. Install dependencies 

`pip install -r requirements.txt`

### 4. Download the MediaPipe model file 
(should be included in the repo)
To download:
`face_landmarker_v2_with_blendshapes.task`

### 5. To run the command-line pipeline 
Run the following command:
`python test_clustering.py`

- This reads all images from a `faces/` folder in the project root
- Saves aligned 125x125 faces crops to `output_faces/`
- Clusters faces by identity and opens a matplotlib grid showing the results
- Prints a summary of identities to the terminal 

### 6. Run the GUI
Run the following command:
`python gui.py`

- Opens a two-panel tkinter window 
- Single Image: 
    - file picker 
    - faces are detected
    - annotated results shown side by side 
- Bulk processing:
    - folder picker
    - process all images on a background thread
    - saves the cropped images to `Processed_Images/`
    - opens a cluster grid plot

### Summary of the files 
`feature_extraction.py` contains the main pipeleine with MediaPipe detection, the InsightFace embedding and DBSCAN clustering. 
`gui.py` has teh tkinter GUI for both single image and bulk processing modes.
`test_clustering.py` is the CLI entry point to run the full pipeline on the `faces/` folder.
`prepreocessing.py` uses CLAHE to contrast the enhancement for dark/low light images.
`skin_segmentation.py` uses a YCrCb skin mask to help filter face detections.
`requirements.txt` contains the python dependencies.