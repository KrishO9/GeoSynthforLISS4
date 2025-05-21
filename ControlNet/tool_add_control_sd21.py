import sys
import os

assert len(sys.argv) == 3, "Args are wrong."

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), "Input model does not exist."
assert not os.path.exists(output_path), "Output filename already exists."
assert os.path.exists(os.path.dirname(output_path)), "Output path is not valid."

import torch

import os
import sys
print(f"--- tool_add_control_sd21.py ---")
print(f"__name__: {__name__}")
print(f"__package__: {__package__}")
print(f"os.getcwd(): {os.getcwd()}")
print(f"sys.path: {sys.path}")
print(f"Attempting to import '.share'")
from .share import *
print(f"Successfully imported share from tool_add_control_sd21.py")


from .cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ""
    p = name[: len(parent_name)]
    if p != parent_name:
        return False, ""
    return True, name[len(parent_name) :]

#A new, "scratch" ControlNet model is created. Its architecture is defined by the cldm_v21.yaml configuration file
model = create_model(config_path="ControlNet/models/cldm_v21.yaml")

pretrained_weights = torch.load(input_path,weights_only=False)
if "state_dict" in pretrained_weights:
    pretrained_weights = pretrained_weights["state_dict"]

# Gets the state dictionary (layer names and their initial random weights) of the newly created ControlNet model.
scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys(): 
    is_control, name = get_node_name(k, "control_") # Checks if the layer k is a ControlNet-specific layer (i.e., its name starts with "control_").
# ControlNet's control-specific layers are initialized by copying weights from corresponding layers in the base SD model's diffusion U-Net. For example, a layer control_input_block_X in ControlNet will try to get its initial weights from model.diffusion_input_block_X of the SD2.1 model.
    if is_control:
        copy_k = "model.diffusion_" + name
    else:
        copy_k = k
#If the corresponding layer (copy_k) exists in the loaded SD2.1 weights.
    if copy_k in pretrained_weights:
#The weights from the SD2.1 model are copied to the target_dict for the current ControlNet layer k
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
#The layer in the ControlNet model keeps its initial (random or zero, depending on create_model) weights.
        target_dict[k] = scratch_dict[k].clone()
        print(f"These weights are newly added: {k}")

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print("Done.")
