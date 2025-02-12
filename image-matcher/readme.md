# Text-to-Image Matcher

A Streamlit application that uses Qdrant vector database and CLIP model for text-to-image matching.

## Demo

![Demo of the Text-to-Image Matcher](demo-visual.gif)

## Features

- Upload multiple images through a simple web interface
- Search images using natural language queries
- Real-time similarity matching using CLIP embeddings
- Persistent storage using local Qdrant instance
- Visual results display with similarity scores

## Prerequisites

- Python 3.8+
- Qdrant running locally on port 6333
- Docker (optional, for running Qdrant)

## Installation

1. Clone this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Qdrant (if not already running):
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload images using the sidebar uploader

4. Enter text queries in the search box to find similar images

## Project Structure

```
.
├── README.md
├── requirements.txt
└── app.py
```

## How It Works

1. **Image Processing**:
   - Images are uploaded through the Streamlit interface
   - CLIP model encodes images into vector embeddings
   - Vectors are stored in Qdrant along with the image data

2. **Text Search**:
   - User enters a text query
   - CLIP model converts text to vector embedding
   - Qdrant performs similarity search
   - Most similar images are displayed with scores

## Technical Details

- Uses CLIP ViT-B-32 model for encoding both images and text
- Vector dimension: 512
- Distance metric: Cosine similarity
- Images are stored as base64 encoded strings in Qdrant payloads

## Configuration

The app connects to Qdrant using these default settings:
- Host: localhost
- Port: 6333

To modify these settings, update the `init_qdrant()` function in `app.py`.

## Limitations

- Currently only supports JPG, JPEG, and PNG image formats
- In-memory image processing may be limited by available RAM
- Search results are limited to top 5 matches

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

[Your chosen license]