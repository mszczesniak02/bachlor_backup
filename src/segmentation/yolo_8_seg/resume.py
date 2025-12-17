import sys
import os
from ultralytics import YOLO

# Constant path to the model checkpoint to resume
# Update this path to point to your specific 'last.pt' file
RESUME_MODEL_PATH = r"/home/krzeslaav/Projects/bachlor/model_tests/FULL_DATASET/yolo_big/runs/segment/yolov8m_crack_seg/weights/last.pt"

# Add project root to sys.path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


def resume_training():
    """
    Resumes training from the checkpoint defined in RESUME_MODEL_PATH.
    """
    checkpoint_path = RESUME_MODEL_PATH

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please check the RESUME_MODEL_PATH constant in the script.")
        return

    print(f"Resuming training from: {checkpoint_path}")

    # Initialize model from checkpoint
    model = YOLO(checkpoint_path)

    # Resume training
    # resume=True automatically loads the state from the checkpoint and continues
    results = model.train(resume=True)

    print("Training resumed and completed.")


if __name__ == "__main__":
    resume_training()
