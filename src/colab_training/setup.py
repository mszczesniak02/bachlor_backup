# run through paths and change them to their coresponding colab alternatives,
# set devices to CUDA's
import os
import sys
# from google.colab import drive
import glob
from pathlib import Path


def make_dirs():
    # make dirs for models & logs
    dirs = [
        "/content/models",

        "/content/models/segmentation",
        "/content/models/classification",
        "/content/models/autoencoder",

        "/content/models/segmentation/unet",
        "/content/models/segmentation/segformer",
        "/content/models/classification/efficienet",
        "/content/models/classification/convnext",

        "/content/models_log",

        "/content/models_log/segmentation",
        "/content/models_log/segmentation/unet",
        "/content/models_log/segmentation/segformer",

        "/content/models_log/classification",
        "/content/models_log/classification/efficienet",
        "/content/models_log/classification/convnext",
        "/content/models_log/autoencoder",
    ]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


# def mount_drive():

#     drive.mount('/content/drive')
#     source = '/content/drive/MyDrive/datasets/multi'  # Na Drive
#     destination = '/content/datasets/multi'            # Link lokalny

#     os.makedirs('/content/datasets', exist_ok=True)

#     if os.path.exists(destination):
#         if os.path.islink(destination):
#             os.unlink(destination)
#         else:
#             import shutil
#             shutil.rmtree(destination)

#     os.symlink(source, destination)

#     print(f"Link created!")
#     print(f"  {destination} -> {source}")


def set_colab(path: str, on_colab: bool):
    dir = Path(path)
    result = list(dir.rglob("hparams.[pP][yY]"))
    for r in result:
        with open(r, "r+") as f:
            pos = f.tell()
            line = f.readline()
            if on_colab:
                if line == "ON_COLAB = False\n":
                    f.seek(pos)
                    f.write("ON_COLAB = True\n")
            else:
                if line == "ON_COLAB = True\n":
                    f.seek(pos)
                    f.write("ON_COLAB = False\n")


def main():

    # make_dirs()
    # mount_drive()
    set_colab(path="..", on_colab=False)


main()

# !cd bachlor_backup/src/autoencoder && python train.py
