import os
import random
import torch
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from torch import nn
from PIL import Image
from transformers import BertTokenizer, BertModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Directory containing images
IMAGE_DIRECTORY = "images"

# Load BERT model and tokenizer for text embedding
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = BertModel.from_pretrained("bert-base-uncased")

# Load a pre-trained ResNet-18 model and modify it
resnet_model = models.resnet18(pretrained=True)

# Get the number of input features for the fully connected layer
num_ftrs = resnet_model.fc.in_features

# Remove the final fully connected layer to get the feature vector
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])

# Add a linear layer to map the feature vector to 768 dimensions (same as BERT's output)
image_embedding_layer = nn.Linear(num_ftrs, 768)

# Set the models to evaluation mode
text_model.eval()
resnet_model.eval()
image_embedding_layer.eval()

# Define image preprocessing for ResNet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper function to compute text embeddings using BERT
def get_text_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Helper function to compute image embeddings using modified ResNet
def get_image_embedding(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        # Extract feature vector from ResNet
        features = resnet_model(image)
        # Apply the linear layer to project to 768 dimensions
        image_embedding = image_embedding_layer(features.squeeze()).numpy()
    return image_embedding

# Helper function to find an image based on cosine similarity
def find_image_for_text(text: str):
    # Get the text embedding
    text_embedding = get_text_embedding(text)
    
    # List all image files in the directory
    image_files = [f for f in os.listdir(IMAGE_DIRECTORY) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        return None
    
    # Compute embeddings for all images
    image_embeddings = []
    image_paths = []
    
    for image_file in image_files:
        image_path = os.path.join(IMAGE_DIRECTORY, image_file)
        image_embedding = get_image_embedding(image_path)
        image_embeddings.append(image_embedding)
        image_paths.append(image_path)
    
    # Compute cosine similarity between text embedding and image embeddings
    similarities = cosine_similarity([text_embedding], image_embeddings)[0]
    
    # Find the index of the most similar image
    best_match_index = np.argmax(similarities)
    return image_paths[best_match_index]

@app.route('/')
def home():
    return jsonify({"message": "Send a POST request to /get_image with plain text as the body to get an image."})

@app.route('/get_image', methods=['POST'])
def get_image():
    # Extract the plain text from the request body
    text = request.data.decode('utf-8').strip()
    
    if not text:
        return jsonify({"error": "Text input is required"}), 400
    
    # Find the most similar image based on cosine similarity
    image_path = find_image_for_text(text)
    
    if not image_path:
        return jsonify({"detail": "Image not found"}), 404
    
    # Return the image as a file response
    return send_file(image_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
