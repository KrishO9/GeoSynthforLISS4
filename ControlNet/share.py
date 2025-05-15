import os
import sys
print(f"--- share.py ---")
print(f"__name__: {__name__}")
print(f"__package__: {__package__}")
print(f"os.getcwd(): {os.getcwd()}")
print(f"sys.path: {sys.path}")
print(f"Attempting to import '.config' from within {__package__ if __package__ else 'an unknown package'}")
from . import config # The problematic line
print(f"Successfully imported config from share.py")

from .cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if config.save_memory:
    enable_sliced_attention()
