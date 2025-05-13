from transformers import pipeline
from PIL import Image
from tqdm import tqdm
import json

files = json.load(open("file_list.json"))
captioner = pipeline("image-text-to-text", model="unsloth/llava-1.5-7b-hf-bnb-4bit")

d = {}

for i in tqdm(range(len(files))):
    image = Image.open(files[i]["target"])
    d[files[i]["target"]] = (
        captioner(
            image,
            prompt="USER: <image>\Describe the contents of the image?\nASSISTANT:",
            generate_kwargs={"max_new_tokens": 76},
        )[0]["generated_text"]
        .split("ASSISTANT:")[-1]
        .strip()
    )
    json.dump(d, open("captions.json", "w"))
