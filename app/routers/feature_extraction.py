from fastapi import APIRouter, Depends, HTTPException
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import faiss
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import psycopg2
from typing import List

# Load a pre-trained model (ResNet50)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()


router = APIRouter(
    prefix="/feature_extract",
    tags=["feature_extract"],
    responses={404: {"description": "Not found"}},
)

fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}


@router.get("/")
async def read_items():
    return fake_items_db


# Preprocess image for the model
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Generate embedding from image
def generate_embedding(image: Image.Image):
    image = preprocess_image(image)
    with torch.no_grad():
        embedding = model(image)  # Forward pass through the model
    return embedding.squeeze().numpy()  # Convert to numpy array

# Setup FAISS index
def create_faiss_index():
    # Fetch embeddings from the database
    embeddings, image_ids = fetch_embeddings_from_db()

    # Initialize FAISS index
    d = embeddings.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(d)  # L2 distance index (Euclidean)
    index.add(embeddings)  # Add vectors to FAISS index
    return index, image_ids

# Fetch image embeddings from PostgreSQL
def fetch_embeddings_from_db():
    conn = psycopg2.connect(
        dbname="your_dbname", user="your_user", password="your_password", host="localhost"
    )
    cur = conn.cursor()
    cur.execute("SELECT image_id, embedding FROM image_embeddings")
    rows = cur.fetchall()
    embeddings = np.array([row[1] for row in rows], dtype=np.float32)
    image_ids = [row[0] for row in rows]
    cur.close()
    conn.close()
    return embeddings, image_ids

# Search for similar images using FAISS
def search_similar_images(query_embedding: np.ndarray, k=3):
    index, image_ids = create_faiss_index()
    query_embedding = query_embedding.astype(np.float32).reshape(1, -1)  # Reshape for FAISS
    distances, indices = index.search(query_embedding, k)  # Search for similar images
    
    # Collect and return similar images based on indices
    similar_images = [(image_ids[i], distances[0][i]) for i in indices[0]]
    return similar_images

# # FastAPI endpoint for uploading an image and getting similar images
# @app.post("/search_images/")
# async def search_images(file: UploadFile = File(...), k: int = 3):
#     # Read the uploaded image
#     image_bytes = await file.read()
#     image = Image.open(BytesIO(image_bytes))

#     # Generate the embedding for the uploaded image
#     query_embedding = generate_embedding(image)

#     # Perform the search
#     similar_images = search_similar_images(query_embedding, k)

#     # Return the search results (image IDs and distances)
#     return {"similar_images": similar_images}



