# The Annotation Garden Project

🌐 **[View Live Dashboard](https://neuromechanist.github.io/image-annotation)**

A VLM-based image annotation system for Natural Scene Dataset (NSD) using multiple language models.

## Overview

This system provides a comprehensive solution for annotating images using various Vision-Language Models (VLMs) including OLLAMA, OpenAI, and Anthropic. It processes images from the Natural Scene Dataset and generates structured annotations that can be used for research and analysis.

## Features

- Multi-model support (OLLAMA, OpenAI GPT-4V, Anthropic Claude)
- Batch processing with real-time progress tracking
- Structured JSON output with schema validation
- Web-based interface for visualization and interaction
- Database storage for efficient querying of 25k+ annotations
- Export functionality for research analysis
- Comprehensive metrics tracking (tokens, speed, timing)
- Stateless processing (fresh context per prompt)
- JSON extraction from markdown-wrapped responses

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OLLAMA (for local models)
- API keys for OpenAI/Anthropic (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-annotation.git
cd image-annotation
```

2. Create and activate a conda environment:
```bash
conda create -n image-annotation python=3.11
conda activate image-annotation
```

3. Install Python dependencies:
```bash
pip install -e .
```

4. Install frontend dependencies:
```bash
cd frontend
npm install
```

## Frontend Dashboard

The project includes a modern web dashboard for browsing and analyzing annotations.

### Quick Start Frontend

```bash
cd frontend

# Install dependencies
npm install

# Copy assets and run development server
npm run copy-assets
npm run dev
```

Visit http://localhost:3000 to see the dashboard.

### Deploy to GitHub Pages

```bash
cd frontend
npm run deploy
```

The dashboard will be available at: https://neuromechanist.github.io/image-annotation

See [frontend/README.md](frontend/README.md) for detailed frontend documentation.

### Configuration

1. Copy the example configuration:
```bash
cp config/config.example.json config/config.json
```

2. Edit `config/config.json` to add your API keys and model preferences

3. Set up NSD images path in configuration

### Usage

### Using the VLM Service Directly

For quick testing or direct VLM annotation without the full stack:

1. Start OLLAMA (if using local models):
```bash
ollama serve
```

2. Run the VLM service test:
```bash
python -m image_annotation.services.vlm_service
```

This will process a sample image with test prompts and save results to `annotations/test/`.

### Using the Full Application

1. Start the backend server:
```bash
uvicorn src.image_annotation.api.main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Open your browser to http://localhost:3000

## Project Structure

```
image-annotation/
├── src/
│   └── image_annotation/
│       ├── api/           # FastAPI endpoints
│       ├── core/          # Core business logic
│       ├── models/        # Data models and schemas
│       ├── services/      # VLM services and adapters
│       └── utils/         # Utility functions
├── frontend/              # Next.js frontend
├── tests/                 # Test suite
├── docs/                  # Documentation
├── config/                # Configuration files
└── scripts/               # Utility scripts
```

## VLM Service API

### Programmatic Usage

```python
from image_annotation.services.vlm_service import VLMService, VLMPrompt
from pathlib import Path

# Initialize service
service = VLMService(
    model="gemma3:4b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Define prompts
prompts = [
    VLMPrompt(
        id="describe",
        text="Describe this image",
        expected_format="text"
    ),
    VLMPrompt(
        id="objects",
        text="List objects in JSON format",
        expected_format="json"
    )
]

# Process images
image_paths = [Path("path/to/image.jpg")]
results = service.process_batch(
    image_paths=image_paths,
    prompts=prompts,
    models=["gemma3:4b", "llava:latest"]
)

# Access results with metrics
for result in results:
    print(f"Model: {result.model}")
    print(f"Tokens used: {result.token_metrics.total_tokens}")
    print(f"Speed: {result.performance_metrics.tokens_per_second} tokens/sec")
    print(f"Response: {result.response}")
    
    # For JSON responses, access parsed data directly
    if result.response_format == "json" and result.response_data:
        print(f"Parsed data: {result.response_data}")
```

## Development

### Running Tests

```bash
pytest tests/ --cov
```

### Code Formatting

```bash
ruff check --fix . && ruff format .
```

### Building for Production

```bash
# Backend
python -m build

# Frontend
cd frontend
npm run build
```

## API Documentation

Once the server is running, visit http://localhost:8000/docs for the interactive API documentation.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Natural Scene Dataset (NSD) team
- LangChain community
- OLLAMA developers