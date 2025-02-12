import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    try:
        client = QdrantClient("localhost", port=6333)
        
        # Create collection if it doesn't exist
        try:
            client.get_collection("text_collection")
        except:
            client.create_collection(
                collection_name="text_collection",
                vectors_config=models.VectorParams(
                    size=384,  # Vector size for all-MiniLM-L6-v2
                    distance=models.Distance.COSINE
                )
            )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {str(e)}")
        st.info("Make sure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`")
        return None

# Get embeddings for text
def get_embedding(text: str, model) -> list[float]:
    return model.encode(text).tolist()

# Initialize app
st.title("Qdrant Vector Database Demo")
st.write("Add text entries and search for similar content using vector similarity!")

# Initialize resources
model = load_model()
client = init_qdrant()

if client is None:
    st.stop()

# Sidebar for adding new entries
st.sidebar.header("Add New Entry")
new_text = st.sidebar.text_area("Enter text")
add_button = st.sidebar.button("Add to Database")

if add_button and new_text:
    try:
        # Generate embedding
        embedding = get_embedding(new_text, model)
        
        # Add to Qdrant
        point_id = np.random.randint(0, 1000000)
        client.upsert(
            collection_name="text_collection",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": new_text}
                )
            ]
        )
        st.sidebar.success(f"Entry added successfully! (ID: {point_id})")
    except Exception as e:
        st.sidebar.error(f"Failed to add entry: {str(e)}")

# Main area for searching
st.header("Search Similar Texts")
search_text = st.text_input("Enter text to search")
top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
search_button = st.button("Search")

if search_button and search_text:
    try:
        # Generate embedding for search text
        search_embedding = get_embedding(search_text, model)
        
        # Search in Qdrant
        search_results = client.search(
            collection_name="text_collection",
            query_vector=search_embedding,
            limit=top_k
        )
        
        # Display results
        if search_results:
            st.subheader("Search Results")
            for i, result in enumerate(search_results, 1):
                with st.expander(f"Result {i} (Similarity: {result.score:.4f})"):
                    st.write(result.payload["text"])
        else:
            st.info("No results found. Try adding some entries first!")
    except Exception as e:
        st.error(f"Search failed: {str(e)}")

# Display collection info
try:
    points_count = client.count(
        collection_name="text_collection",
        exact=True
    ).count
    st.header("Collection Statistics")
    st.write(f"Number of entries: {points_count}")
except Exception as e:
    st.error(f"Failed to get collection count: {str(e)}")

# Add option to clear collection
if st.button("Clear Collection"):
    try:
        client.delete_collection("text_collection")
        client.create_collection(
            collection_name="text_collection",
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
        st.success("Collection cleared successfully!")
    except Exception as e:
        st.error(f"Failed to clear collection: {str(e)}")

# Add example data button
if st.button("Add Example Data"):
    example_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a versatile programming language",
        "Neural networks are inspired by biological neurons",
        "Vector databases are useful for similarity search"
    ]
    try:
        for text in example_texts:
            embedding = get_embedding(text, model)
            client.upsert(
                collection_name="text_collection",
                points=[
                    models.PointStruct(
                        id=np.random.randint(0, 1000000),
                        vector=embedding,
                        payload={"text": text}
                    )
                ]
            )
        st.success(f"Added {len(example_texts)} example entries!")
    except Exception as e:
        st.error(f"Failed to add example data: {str(e)}")