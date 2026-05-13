from roboflow import Roboflow
import os

# Authenticate with your API key
rf = Roboflow(api_key="RS2bcmFwLZ0T6ZPHONoQ")  # Replace with your API key

# Access the dataset
# the link of the dataset: https://universe.roboflow.com/underwater-plastic-detection-1btop/underwater-plastic-segmentation
project = rf.workspace("underwater-plastic-detection-1btop").project("underwater-plastic-segmentation")
dataset = project.version("2").download("yolov8")  # Download the dataset

# Define the original folder name and the new folder name
original_folder_name = "Underwater-Plastic-Segmentation-2"  # Replace with the actual folder name if different
new_folder_name = "segmentation_dataset-1"

# Rename the folder using Python
if os.path.exists(original_folder_name):
    os.rename(original_folder_name, new_folder_name)
    print(f"Folder renamed from '{original_folder_name}' to '{new_folder_name}'.")
else:
    print(f"Folder '{original_folder_name}' does not exist. Check the download path.")