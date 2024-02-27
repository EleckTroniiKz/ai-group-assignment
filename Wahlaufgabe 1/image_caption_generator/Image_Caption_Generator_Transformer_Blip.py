import os
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pandas as pd

# Code inspiriert durch https://huggingface.co/Salesforce/blip-image-captioning-large
# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Define the directory containing the local images
image_dir = "image_data/FiftyArtImages"

# Get the list of image filenames in the directory
image_filenames = os.listdir(image_dir)

# Process images in batches
batch_size = 2  # Define the batch size
outputs = []

# Measure total time
start_time = time.time()

# Iterate over images in batches
for i in range(0, len(image_filenames), batch_size):
    batch_images = []
    batch_filenames = image_filenames[i:i + batch_size]

    # Load and preprocess images in the batch
    for filename in batch_filenames:
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        batch_images.append(image)

    # Encode images using the BLIP processor
    inputs = processor(batch_images, return_tensors="pt")

    # Perform caption generation
    with torch.no_grad():
        out = model.generate(**inputs)

    # Decode the generated captions
    captions = [processor.decode(output, skip_special_tokens=True) for output in out]
    outputs.extend(captions)

# Calculate total time taken
total_time = time.time() - start_time

# Create a DataFrame with the results
results_df = pd.DataFrame({"Image Filename": image_filenames, "Caption": outputs})

# Save the DataFrame to a CSV file
results_df.to_csv("result.csv", index=False)

# Print the total time
print(f"Total time taken for processing {len(image_filenames)} images: {total_time:.2f} seconds")
