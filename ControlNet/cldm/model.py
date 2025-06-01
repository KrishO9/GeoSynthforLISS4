import os
import torch

from omegaconf import OmegaConf
from ..ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get("state_dict", d)


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch

        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(
            torch.load(ckpt_path, map_location=torch.device(location), weights_only=True)
        )
    state_dict = get_state_dict(state_dict)
    print(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict


# def create_model(config_path):
#     config = OmegaConf.load(config_path)
#     model = instantiate_from_config(config.model).cpu()
#     print(f"Loaded model config from [{config_path}]")
#     return model

def create_model(config_path):
    print(f"--- In create_model (ControlNet/cldm/model.py) ---")
    abs_config_path = os.path.abspath(config_path)
    print(f"Attempting to load OmegaConf from absolute config_path: {abs_config_path}")
    if not os.path.exists(abs_config_path):
        print(f"ERROR: Config file not found at {abs_config_path}")
        raise FileNotFoundError(f"Config file not found: {abs_config_path}")
        
    config = OmegaConf.load(config_path) # Loads the ENTIRE YAML

    print("\n=== Loaded Full Configuration (from cldm_v21.yaml) ===")
    print(OmegaConf.to_yaml(config)) # Prints the whole loaded config

    # The 'model' section is passed to instantiate_from_config
    model_config_section = config.model 
    
    print("\n=== 'model' Section Being Passed to instantiate_from_config ===")
    print(f"Target class for main model: {model_config_section.target}")
    print(OmegaConf.to_yaml(model_config_section.params)) # Print all params for the main model

    # Specifically print sub-configurations that define large components
    if hasattr(model_config_section.params, 'control_stage_config'):
        print("\n--- control_stage_config ---")
        print(f"Target: {model_config_section.params.control_stage_config.target}")
        print(OmegaConf.to_yaml(model_config_section.params.control_stage_config.params))

    if hasattr(model_config_section.params, 'unet_config'):
        print("\n--- unet_config ---")
        print(f"Target: {model_config_section.params.unet_config.target}")
        print(OmegaConf.to_yaml(model_config_section.params.unet_config.params))

    if hasattr(model_config_section.params, 'first_stage_config'):
        print("\n--- first_stage_config (VAE) ---")
        print(f"Target: {model_config_section.params.first_stage_config.target}")
        # VAE params can be verbose, print ddconfig specifically if needed
        if hasattr(model_config_section.params.first_stage_config.params, 'ddconfig'):
             print(OmegaConf.to_yaml(model_config_section.params.first_stage_config.params.ddconfig))
        else:
             print(OmegaConf.to_yaml(model_config_section.params.first_stage_config.params))


    if hasattr(model_config_section.params, 'cond_stage_config'):
        print("\n--- cond_stage_config (Text Encoder) ---")
        print(f"Target: {model_config_section.params.cond_stage_config.target}")
        print(OmegaConf.to_yaml(model_config_section.params.cond_stage_config.params))

    # The original script might modify ckpt_path here, ensure it's handled if necessary
    # This part was in your original `create_model` from the repo, but `ckpt` is not passed
    # if "params" not in config.model:
    #     config.model.params = dict()
    # if isinstance(config.model.params, DictConfig): # Check if it's a DictConfig
    #    config.model.params.ckpt_path = None # os.path.abspath(ckpt) # ckpt is not defined here
    # else: # If it's a regular dict
    #    if 'params' not in config.model:
    #        config.model['params'] = {}
    #    config.model['params']['ckpt_path'] = None


    print("\nInstantiating model from the above configuration...")
    model = instantiate_from_config(model_config_section).cpu()
    print("Model instantiated successfully on CPU.")
    return model
