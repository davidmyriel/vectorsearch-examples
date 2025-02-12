import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import io
import base64
import uuid

# Initialize Streamlit app
st.title("Text-to-Image Matcher")

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    return QdrantClient(host="localhost", port=6333)  # Connect to local Qdrant server

# Initialize SBERT model
@st.cache_resource
def init_model():
    return SentenceTransformer('clip-ViT-B-32')

# Create collection if it doesn't exist
def create_collection(client, collection_name="images"):
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "image_vector": VectorParams(size=512, distance=Distance.COSINE),
                "text_vector": VectorParams(size=512, distance=Distance.COSINE)
            }
        )

# Function to process and upload image
def process_image(client, model, image_file, collection_name="images"):
    # Read and preprocess image
    image = Image.open(image_file).convert('RGB')
    image_input = model.encode(image)
    
    # Create point using PointStruct
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "image_vector": image_input.tolist(),
            "text_vector": [0.0] * 512  # Placeholder for text vector
        },
        payload={
            "image_data": base64.b64encode(image_file.getvalue()).decode()
        }
    )
    
    # Upload to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=[point]
    )

# Function to search images by text
def search_images(client, model, text_query, collection_name="images", limit=5):
    # Encode text query
    text_vector = model.encode(text_query)
    
    # Search in Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=("image_vector", text_vector.tolist()),
        limit=limit
    )
    
    return results

# Initialize components
client = init_qdrant()
model = init_model()
create_collection(client)

# Sidebar for uploading images
st.sidebar.header("Upload Images")
uploaded_files = st.sidebar.file_uploader(
    "Choose images", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Process uploaded images
if uploaded_files:
    for file in uploaded_files:
        process_image(client, model, file)
    st.sidebar.success(f"Uploaded {len(uploaded_files)} images!")

# Main area for searching
st.header("Search Images")
text_query = st.text_input("Enter your text query")

if text_query:
    results = search_images(client, model, text_query)
    
    if results:
        st.subheader("Results")
        cols = st.columns(min(len(results), 3))
        
        for idx, result in enumerate(results):
            col = cols[idx % 3]
            with col:
                image_data = base64.b64decode(result.payload["image_data"])
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption=f"Score: {result.score:.2f}")
    else:
        st.info("No matching images found.")