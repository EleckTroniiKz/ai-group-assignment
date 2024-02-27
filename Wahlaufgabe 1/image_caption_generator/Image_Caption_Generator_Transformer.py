# 1. Load the pre-trained model from the model hub
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

model_name = "nlpconnect/vit-gpt2-image-captioning"

# load model
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Load the image for which the caption is to be generated
# img_name = "flickr_data.jpg"
print("Model loaded")
img_name = "35.png"
img_path = "image_data/FiftyArtImages/" + img_name
# img = Image.open(img_name)
img = Image.open(img_path)
if img.mode != 'RGB':
    img = img.convert(mode="RGB")

pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

max_length = 128
num_beams = 4

# get model prediction
print("Pixel preprocessed")
output_ids = model.generate(pixel_values, num_beams=num_beams, max_length=max_length)

# decode the generated prediction
preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(preds)
