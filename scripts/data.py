import json
import cv2
import numpy as np
import os # Import os module
from torch.utils.data import Dataset as TorchDataset # Use an alias to avoid confusion

class Dataset(TorchDataset): # Inherit from the aliased TorchDataset
    def __init__(self, prompt_path, location_embeds_path, image_base_dir="/kaggle/input/geosynth64/"):
        # It's safer to open and close the file properly
        with open(prompt_path, "rt") as f:
            self.data = json.load(f)
        self.loc_embeds = np.load(location_embeds_path)
        self.image_base_dir = image_base_dir

        print(f"Dataset initialized. Image base directory: {self.image_base_dir}")
        if not os.path.isdir(self.image_base_dir):
            print(f"WARNING: Image base directory '{self.image_base_dir}' does not exist or is not a directory.")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # These are filenames/paths *relative to the image_base_dir*
        # e.g., item["source"] might be "osm_aligned_patches/osm_ref_patch_00363.png"
        # OR if your JSON *already* contains the problematic "..//kaggle/input/..." paths,
        # you'll need to clean them first (see note below).
        relative_source_path = item["source"]
        relative_target_path = item["target"]
        prompt = item["prompt"]
        loc_ind = item["class"] # Make sure this key 'class' is correct for your JSON

        # --- Construct absolute paths ---
        # This is the crucial change:
        absolute_source_filename = os.path.join(self.image_base_dir, relative_source_path)
        absolute_target_filename = os.path.join(self.image_base_dir, relative_target_path)

        # --- Add debugging prints ---
        # print(f"Attempting to load SOURCE: {absolute_source_filename}")
        # print(f"Attempting to load TARGET: {absolute_target_filename}")
        # --- End debugging prints ---

        source = cv2.imread(absolute_source_filename)
        target = cv2.imread(absolute_target_filename)

        # --- Add checks for successful image loading ---
        if source is None:
            error_msg = f"Failed to read SOURCE image at path: {absolute_source_filename}. Original relative path: {relative_source_path}"
            print(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg) # Fail loudly if image not found
        if target is None:
            error_msg = f"Failed to read TARGET image at path: {absolute_target_filename}. Original relative path: {relative_target_path}"
            print(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg) # Fail loudly if image not found
        # --- End checks ---

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(
            jpg=target, txt=prompt, hint=source, location=self.loc_embeds[loc_ind]
        )