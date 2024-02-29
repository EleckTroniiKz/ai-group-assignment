import os
import pandas as pd
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class ModelTester:
    def __init__(self, model_name, image_dir, max_length, num_beams_values):
        self.model_name = model_name
        self.image_dir = image_dir
        self.max_length = max_length
        self.num_beams_values = num_beams_values
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_images(self):
        feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
        images = []
        image_paths = []
        for filename in os.listdir(self.image_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(self.image_dir, filename)
                image = Image.open(image_path).convert("RGB")
                image = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
                images.append(image)
                image_paths.append(image_path)
        return images, image_paths

    def generate_captions(self, images, image_paths, num_beams):
        results = []
        captions = []
        with torch.no_grad():
            for idx, image in enumerate(tqdm(images, desc=f"Generating captions with num_beams={num_beams}")):
                output_ids = self.model.generate(image, num_beams=num_beams, max_length=self.max_length)
                preds = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                results.append({"Image Filename": image_paths[idx], "Caption": preds})
                captions.append(preds)
        return results, captions

    def calculate_similarity(self, captions):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(captions)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix

    def save_results_to_csv(self, results, csv_filename):
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

    def run_tests(self):
        for num_beams in self.num_beams_values:
            images, image_paths = self.preprocess_images()
            results, captions = self.generate_captions(images, image_paths, num_beams)
            similarity_matrix = self.calculate_similarity(captions)
            csv_filename = f"{self.model_name}_num_beams_{num_beams}_results.csv"
            self.save_results_to_csv(results, csv_filename)
            print(f"Similarity matrix for num_beams={num_beams}:\n{similarity_matrix}")


# Define parameters
model_name = "nlpconnect/vit-gpt2-image-captioning"
image_dir = "test_data"
max_length = 128
num_beams_values = [5, 10, 15]  # Different values for num_beams

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ModelTester
tester = ModelTester(model_name, image_dir, max_length, num_beams_values)

# Run tests
tester.run_tests()
