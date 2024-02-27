import os
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def createModell(batch_size, max_length, num_beams, temperature):
    # Load the BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Define the directory containing the local images
    image_dir = "test_data"

    # Get the list of image filenames in the directory
    image_filenames = os.listdir(image_dir)

    # Process images in batches
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
            out = model.generate(**inputs, max_length=max_length, num_beams=num_beams, temperature=temperature)

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

    # Calculate total time taken
    total_time = time.time() - start_time

    # Initialize the Tf-idf vectorizer and apply it to the captions
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(outputs)

    # Get the feature names (in this case the words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create a DataFrame for the Tf-idf matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Save the Tf-idf matrix to a CSV file
    tfidf_df.to_csv(f"tfidf_matrix_{batch_size}_{max_length}_{num_beams}_{temperature}.csv", index=False)

    # Print the total time
    print(
        f"Total time taken for processing {len(image_filenames)} images with batch_size={batch_size}, max_length={max_length}, num_beams={num_beams}, temperature={temperature}: {total_time:.2f} seconds")


def run_tests():
    # Define the parameters to test
    batch_sizes = [1]
    max_lengths = [50, 100, 150]
    num_beams_list = [2, 4, 6]
    temperatures = [0.7, 0.8, 0.9]

    # Iterate over all combinations of parameters
    for batch_size in batch_sizes:
        for max_length in max_lengths:
            for num_beams in num_beams_list:
                for temperature in temperatures:
                    # Test the model with these parameters
                    createModell(batch_size, max_length, num_beams, temperature)


run_tests()
