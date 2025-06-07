import os
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
from tqdm import tqdm
from preprocess.config import EXTRACTED_DATASET_PATH

# 1) Configuration
DATA_ROOT  = EXTRACTED_DATASET_PATH
SPLITS     = ['train', 'val', 'test']
LABELS     = ['0', '1']
SEQ_LEN    = 24

# 2) Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 3) Landmark indices for EAR and MAR
LEFT_EYE_IDXS   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS  = [362, 385, 387, 263, 373, 380]
TOP_LIP_IDX     = 13
BOTTOM_LIP_IDX  = 14
LEFT_MOUTH_IDX  = 78
RIGHT_MOUTH_IDX = 308

# 4) Compute EAR and MAR
def compute_ear(landmarks, w, h):
    def ear_for(idxs):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C + 1e-8)
    return (ear_for(LEFT_EYE_IDXS) + ear_for(RIGHT_EYE_IDXS)) / 2.0


def compute_mar(landmarks, w, h):
    pt_top   = np.array((landmarks[TOP_LIP_IDX].x * w, landmarks[TOP_LIP_IDX].y * h))
    pt_bot   = np.array((landmarks[BOTTOM_LIP_IDX].x * w, landmarks[BOTTOM_LIP_IDX].y * h))
    pt_left  = np.array((landmarks[LEFT_MOUTH_IDX].x * w, landmarks[LEFT_MOUTH_IDX].y * h))
    pt_right = np.array((landmarks[RIGHT_MOUTH_IDX].x * w, landmarks[RIGHT_MOUTH_IDX].y * h))
    vert = np.linalg.norm(pt_top - pt_bot)
    horz = np.linalg.norm(pt_left - pt_right)
    return vert / (horz + 1e-8)

# 5) Load features for a split
def load_features(split):
    X_all, y_all = [], []
    split_dir = os.path.join(DATA_ROOT, split)

    tasks = []
    for subfolder in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, subfolder)
        if not os.path.isdir(folder_path):
            continue
        try:
            label = int(subfolder.split('_')[-1])
        except:
            continue
        for img_path in sorted(glob(os.path.join(folder_path, '*.jpg'))):
            tasks.append((subfolder, label, img_path))

    groups = {} 
    for subfolder, label, img_path in tqdm(tasks, desc=f"Extracting {split}", ncols=80):
        folder = os.path.join(split_dir, subfolder)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue

        lm = res.multi_face_landmarks[0].landmark
        ear = compute_ear(lm, w, h)
        try:
            mar = compute_mar(lm, w, h)
        except:
            mar = 0.0

        if subfolder not in groups:
            groups[subfolder] = {'label': label, 'X': [], 'y': []}
        groups[subfolder]['X'].append([ear, mar])
        groups[subfolder]['y'].append(label)

    for info in groups.values():
        Xv = np.array(info['X'], dtype=np.float32)
        yv = np.array(info['y'], dtype=np.int32)
        if len(Xv) >= SEQ_LEN:
            X_all.append(Xv)
            y_all.append(yv)

    return X_all, y_all

# 6) Extract features
X_train, y_train = load_features('train')
X_val,   y_val   = load_features('val')
X_test,  y_test  = load_features('test')

# Convert to object arrays so we can savez heterogeneous lists
X_train = np.array(X_train, dtype=object)
y_train = np.array(y_train, dtype=object)
X_val   = np.array(X_val,   dtype=object)
y_val   = np.array(y_val,   dtype=object)
X_test  = np.array(X_test,  dtype=object)
y_test  = np.array(y_test,  dtype=object)

# Save to a .npz (compressed) with pickled object arrays
np.savez_compressed(
    "features_ear_mar.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test
)
print("✅ Features saved to features_ear_mar.npz")

print("Feature extraction complete:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape,   y_val.shape)
print("Test: ", X_test.shape,  y_test.shape)