import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scripts.data import Dataset
from ControlNet.cldm.logger import ImageLogger
from ControlNet.cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch 
from pytorch_lightning.strategies import DDPStrategy

# Configs
resume_path = "prepared_checkpoint/control_sd21_ini.ckpt"
batch_size = 1
logger_freq = 2000
learning_rate = 1e-5
sd_locked = True
only_mid_control = True

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model("scripts/models/cldm_v21.yaml").cpu()
model.load_state_dict(load_state_dict(resume_path, location="cpu"))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

checkpoint = ModelCheckpoint(
    dirpath=os.path.join("checkpoint", "geosynth"),
    filename="model_geosynth",
    every_n_train_steps=150,
)

# Misc
dataset = Dataset(
    # prompt_path="scripts/prompt_with_location.json",
    # location_embeds_path="scripts/location_embeds.npy",
    prompt_path="/content/drive/MyDrive/Data/prompt_with_location.json",
    location_embeds_path="/content/drive/MyDrive/Data/location_embeds.npy",
)
dataloader = DataLoader(
    dataset,
    num_workers=1,
    batch_size=batch_size,
    shuffle=True,
    persistent_workers=False
)
logger = ImageLogger(batch_frequency=logger_freq)


torch.cuda.empty_cache()
print("CUDA cache emptied.")

trainer = pl.Trainer(
    accelerator="gpu",
    strategy=DDPStrategy(find_unused_parameters=True),
    devices=1,
    precision="16-mixed",
    max_epochs=1,
    max_steps=10,
    callbacks=[logger, checkpoint],
    accumulate_grad_batches=1,
)

# Train!
trainer.fit(model, dataloader)