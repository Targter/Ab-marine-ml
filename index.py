import os
import logging
import torch
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from torch import nn
from PIL import Image
from transformers import DistilBertTokenizer, DistilBertModel
from multiprocessing import Pool

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories and constants
IMAGE_DIR = "images"

# Load DistilBERT for text embeddings
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = DistilBertModel.from_pretrained("distilbert-base-uncased").eval()

# Load ResNet-18 for image embeddings
resnet = models.resnet18(pretrained=True)
image_embedding_layer = nn.Linear(resnet.fc.in_features, 768)
resnet = nn.Sequential(*list(resnet.children())[:-1]).eval()

# Image transformation pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preloaded embeddings and paths
image_embeddings = []
image_paths = []

def get_embedding(data, model, is_text=True):
    """
    Generate embeddings for text or images using the respective model.
    """
    try:
        with torch.no_grad():
            if is_text:
                inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True, max_length=512)
                return model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
            else:
                features = resnet(data.unsqueeze(0))
                return image_embedding_layer(features.squeeze()).numpy()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def preload_images():
    """
    Preload all images in the directory and compute their embeddings.
    """
    global image_embeddings, image_paths
    images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        logger.warning("No images found in the directory.")
        return

    def process_image(img_path):
        try:
            image = image_transform(Image.open(img_path).convert("RGB"))
            embedding = get_embedding(image, resnet, is_text=False)
            return img_path, embedding
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return None, None

    with Pool() as pool:
        results = pool.map(process_image, images)

    for img_path, embedding in results:
        if img_path and embedding is not None:
            image_paths.append(img_path)
            image_embeddings.append(embedding)

def find_best_image(text):
    """
    Find the best matching image for a given text based on cosine similarity.
    """
    if not image_embeddings:
        logger.error("No image embeddings are loaded.")
        return None

    try:
        text_emb = get_embedding(text, text_model)
        similarities = cosine_similarity([text_emb], image_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        logger.info(f"Best match: {image_paths[best_match_idx]} with similarity: {similarities[best_match_idx]}")
        return image_paths[best_match_idx]
    except Exception as e:
        logger.error(f"Error finding best image: {e}")
        return None

@app.route('/')
def home():
    """
    Home route to indicate the service is running.
    """
    return jsonify({"message": "Send a POST request to /get_image with plain text to get a matching image."})

@app.route('/get_image', methods=['POST'])
def get_image():
    """
    Endpoint to handle text input and return the best matching image.
    """
    text = request.data.decode('utf-8').strip()
    if not text:
        return jsonify({"error": "Text input is required"}), 400

    image_path = find_best_image(text)
    if image_path:
        return send_file(image_path, mimetype='image/jpeg')
    return jsonify({"detail": "No matching image found"}), 404

if __name__ == '__main__':
    logger.info("Preloading images...")
    preload_images()
    logger.info("Starting Flask app...")
    app.run(debug=True)
