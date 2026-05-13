import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("surajit651/souvikdataset")

# Define source directory (assuming "SOUVIK" is always the subfolder)
source_dir = os.path.join(path, "SOUVIK")

# Define target directory
target_dir = "mydataset"

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# Move files from "SOUVIK" to "mydataset"
for file in os.listdir(source_dir):
    shutil.move(os.path.join(source_dir, file), target_dir)

print("Dataset moved to:", target_dir)
