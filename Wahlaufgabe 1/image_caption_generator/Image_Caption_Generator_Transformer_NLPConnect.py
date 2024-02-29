import os
import time
import pandas as pd
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Code inspiriert durch https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
# Load pre-trained model
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image directory
image_dir = "image_data/FiftyArtImages"
max_length = 128
num_beams = 15

# Load and preprocess images
images = []
image_paths = []
for filename in os.listdir(image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
        images.append(image)
        image_paths.append(image_path)

# Batch processing with time measurement
results = []
with torch.no_grad():
    for idx, image in enumerate(images):
        start_time = time.time()
        output_ids = model.generate(image, num_beams=num_beams, max_length=max_length)
        end_time = time.time()
        preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({"Image Filename": image_paths[idx], "Caption": preds})
        processing_time = end_time - start_time
        print(f"Image: {image_paths[idx]}, Caption: {preds}")
        print(f"Time taken for processing: {processing_time:.2f} seconds")

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print("Results saved to nlp_connect_result.csv")
