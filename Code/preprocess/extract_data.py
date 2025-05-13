import os
import cv2
import shutil
from glob import glob
import logging
from tqdm import tqdm
from config import RAW_DATASET_PATH, EXTRACTED_DATASET_PATH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Configuration
FRAME_INTERVAL = 2

# Local paths
TRAIN_DIR = os.path.join(RAW_DATASET_PATH, 'Train')
TEST_DIR = os.path.join(RAW_DATASET_PATH, 'Test')
OUTPUT_DIR = EXTRACTED_DATASET_PATH

def rotate_if_needed(frame):
    # Only rotate if width > height (landscape)
    if frame.shape[1] > frame.shape[0]:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# Extract and save frames into a per-video directory
def extract_and_save_grouped_frames(video_path, label, split, subject_name, time_of_day, frame_interval=FRAME_INTERVAL):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0

    # Create a subdirectory for each video based on label
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_folder_name = f"{subject_name}_{time_of_day}_{video_id}_{label}"
    video_output_dir = os.path.join(OUTPUT_DIR, split, video_folder_name)
    os.makedirs(video_output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = rotate_if_needed(frame)
            frame = cv2.resize(frame, (360, 640))
            save_path = os.path.join(video_output_dir, f"frame{frame_id:03d}.jpg")
            cv2.imwrite(save_path, frame)
            frame_id += 1
        count += 1
    cap.release()

# Extract frames from flat video files (e.g., test set)
def extract_from_flat_dir(split_dir, split_name):
    video_files = glob(os.path.join(split_dir, '*.*'))
    for video in tqdm(video_files, desc=f"Videos in {split_name}", unit="video"):
        try:
            label = int(os.path.basename(video).split('.')[0])
            extract_and_save_grouped_frames(
                video, label, split=split_name,
                subject_name='Agung', time_of_day='days'
            )
        except Exception as e:
            logger.error(f"Error processing {video}: {e}")

# Process videos into frames and save into per-video folders
def process_videos_to_grouped_frames():
    for split_dir, split_name in [(TRAIN_DIR, 'train'), (TEST_DIR, 'test')]:
        split_output_dir = os.path.join(OUTPUT_DIR, split_name)
        if os.path.exists(split_output_dir) and os.listdir(split_output_dir):
            logger.info(f"Skipping {split_name} - already contains data.")
            continue

        logger.info(f"Processing {split_name} dataset...")

        if split_name == 'test':
            extract_from_flat_dir(split_dir, split_name)
        else:
            subject_dirs = [subject for subject in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, subject))]

            for subject in tqdm(subject_dirs, desc=f"Subjects in {split_name}", unit="subject"):
                subject_path = os.path.join(split_dir, subject)
                for time_of_day in ['days', 'evenings']:
                    time_path = os.path.join(subject_path, time_of_day)
                    if not os.path.isdir(time_path):
                        continue
                    video_files = glob(os.path.join(time_path, '*'))
                    for video in tqdm(video_files, desc=f"Videos in {time_of_day} ({subject})", unit="video", leave=False):
                        try:
                            label = int(os.path.basename(video).split('.')[0])
                            target_split = 'val' if subject.lower() == 'hanif' else 'train'
                            extract_and_save_grouped_frames(
                                video, label, split=target_split,
                                subject_name=subject, time_of_day=time_of_day
                            )
                        except Exception as e:
                            logger.error(f"Error processing {video}: {e}")

if __name__ == '__main__':
    process_videos_to_grouped_frames()
    print("✅ Frame extraction (grouped by video+label) completed.")
