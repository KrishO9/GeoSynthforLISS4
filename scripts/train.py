import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scripts.data import Dataset # Assuming this is correctly importable
from ControlNet.cldm.logger import ImageLogger # Assuming this is correctly importable
from ControlNet.cldm.model import create_model, load_state_dict # Assuming these are correctly importable
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch # Added for potential torch.cuda.empty_cache()

# Configs
resume_path = "prepared_control_checkpoint/control_sd21_ini.ckpt"
# --- AGGRESSIVE MEMORY SAVING PARAMETER CHANGES ---
batch_size = 1                 # Reduced from 2
logger_freq = 10               # Reduced for more frequent logging in short run
learning_rate = 1e-5           # Kept the same
sd_locked = True
only_mid_control = False
# --- END AGGRESSIVE CHANGES ---


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model("scripts/models/cldm_v21.yaml").cpu()

# Conditional loading of state dict
if os.path.exists(resume_path):
    print(f"Loading state dictionary from: {resume_path}")
    model.load_state_dict(load_state_dict(resume_path, location="cpu"), strict=False) # Added strict=False for robustness
else:
    print(f"Warning: Resume path '{resume_path}' not found. Model will use initial weights.")

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Ensure checkpoint directory exists
checkpoint_dirpath = os.path.join("checkpoint", "geosynth_oom_test") # Use a distinct dir
os.makedirs(checkpoint_dirpath, exist_ok=True)

checkpoint = ModelCheckpoint(
    dirpath=checkpoint_dirpath, # Use defined variable
    filename="model_geosynth_oom_test", # Distinct filename
    every_n_train_steps=5,  # --- AGGRESSIVE CHANGE: Checkpoint very frequently ---
    save_top_k=-1           # Save all checkpoints during this short test
)
# Misc
dataset = Dataset(
    prompt_path="scripts/prompt_with_location.json",
    location_embeds_path="scripts/location_embeds.npy",
)
dataloader = DataLoader(
    dataset,
    num_workers=0,             # --- AGGRESSIVE CHANGE: Reduce num_workers for DDP debugging ---
    batch_size=batch_size,     # Set to 1
    shuffle=True,
    persistent_workers=False   # --- AGGRESSIVE CHANGE: Must be False if num_workers=0 ---
)
logger = ImageLogger(batch_frequency=logger_freq) # logger_freq is already reduced

trainer = pl.Trainer(
    accelerator="gpu",
    # --- STRATEGY AND DEVICES FOR DEBUGGING ---
    # Step 1: Try with 1 GPU first to confirm baseline memory
    devices=1,
    # strategy=None, # No DDP strategy needed for devices=1

    # Step 2: If 1 GPU works, uncomment below and comment out devices=1 and strategy=None
    # devices=2,
    # strategy="ddp", # Start with simpler "ddp", then try "ddp_find_unused_parameters_true" if needed

    precision="16-mixed",      # Already good for memory
    # max_epochs=1,            # Original: This is fine for a short run
    max_steps=10,             # --- AGGRESSIVE CHANGE: Very few steps ---
    callbacks=[logger, checkpoint],
    accumulate_grad_batches=1, # --- AGGRESSIVE CHANGE: No accumulation if batch_size is 1 per GPU ---
    # --- Optional: Further aggressive changes for debugging if still OOM ---
    # limit_train_batches=5,   # Process only 5 batches per epoch
    # log_every_n_steps=1,     # Log every single step
    # gradient_clip_val=0.5,   # Can sometimes help with stability / exploding gradients
)

# Train!
print("Starting training with aggressive memory-saving parameters...")
try:
    trainer.fit(model, dataloader)
    print("Training finished a few steps successfully!")
except Exception as e:
    print(f"An error occurred during training: {e}")
    import traceback
    traceback.print_exc()
    if 'cuda' in str(e).lower() and 'out of memory' in str(e).lower():
        print("CUDA Out of Memory error detected.")
        if torch.cuda.is_available():
            print("Attempting to clear CUDA cache...")
            torch.cuda.empty_cache()
            print("CUDA cache cleared (if supported).")