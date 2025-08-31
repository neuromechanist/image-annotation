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
- **Annotation manipulation tools** for post-processing:
  - Reorder model annotations
  - Remove specific models
  - Filter by token usage
  - Export to CSV for analysis
  - Batch operations on entire datasets

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

## Annotation Tools

The project includes powerful tools for manipulating annotation JSON files. These tools support both single file and batch operations.

### Installation

```bash
pip install -e .
```

### Available Tools

#### 1. Reorder Annotations
Reorder model annotations to a specified order:

```python
from image_annotation.tools import reorder_annotations

# Single file
reorder_annotations("annotation.json", ["model1", "model2", "model3"])

# Directory (batch)
reorder_annotations("annotations/", ["model1", "model2", "model3"])

# Multiple specific files
reorder_annotations([file1, file2], ["model1", "model2", "model3"])
```

#### 2. Remove Model Annotations
Remove specific model annotations from files:

```python
from image_annotation.tools import remove_model

# Remove from single file
remove_model("annotation.json", "unwanted_model")

# Remove from all files in directory
remove_model("annotations/", "unwanted_model")
```

#### 3. Filter by Token Count
Filter annotations based on token usage:

```python
from image_annotation.tools import filter_annotations_by_tokens

# Keep only annotations using less than 1000 tokens
filter_annotations_by_tokens("annotations/", max_tokens=1000)
```

#### 4. Get Statistics
Analyze your annotation dataset:

```python
from image_annotation.tools import get_annotation_stats

stats = get_annotation_stats("annotations/")
print(f"Total files: {stats['files_processed']}")
print(f"Total annotations: {stats['total_annotations']}")
print(f"Models used: {stats['models']}")
```

#### 5. Export to CSV
Export annotations for analysis:

```python
from image_annotation.tools import export_to_csv

export_to_csv("annotations/", "output.csv", include_metrics=True)
```

### Command Line Usage

```bash
# Get statistics
python -m image_annotation.tools.annotation_tools stats annotations/

# Reorder annotations
python -m image_annotation.tools.annotation_tools reorder annotations/

# Remove a model
python -m image_annotation.tools.annotation_tools remove annotations/ model_name

# Export to CSV
python -m image_annotation.tools.annotation_tools export annotations/ output.csv
```

## NSD Annotation Process

### Preparing NSD Images

1. **Download NSD Dataset**: Obtain the Natural Scene Dataset images from the official source
2. **Organize Images**: Place images in a structured directory (e.g., `/path/to/NSD_stimuli/shared1000/`)
3. **Configure Path**: Update the configuration file with your NSD images path

### Running Batch Annotations

#### Step 1: Configure Models and Prompts

Create a configuration file with your desired models and prompts:

```python
models = [
    "qwen2.5vl:7b",
    "qwen2.5vl:32b", 
    "gemma3:4b",
    "gemma3:12b",
    "gemma3:27b",
    "mistral-small3.2:24b"
]

prompts = [
    VLMPrompt(
        id="general_description",
        text="Describe what you see in this image...",
        expected_format="text"
    ),
    VLMPrompt(
        id="foreground_background",
        text="Describe the foreground and background elements...",
        expected_format="text"
    ),
    # Add more prompts as needed
]
```

#### Step 2: Process Images

```python
from image_annotation.services.vlm_service import VLMService
from pathlib import Path

# Initialize service
service = VLMService(model=models[0])

# Get NSD images
image_dir = Path("/path/to/NSD_stimuli/shared1000/")
image_paths = sorted(image_dir.glob("*.png"))

# Process in batches
batch_size = 100
for i in range(0, len(image_paths), batch_size):
    batch = image_paths[i:i+batch_size]
    results = service.process_batch(
        image_paths=batch,
        prompts=prompts,
        models=models
    )
    # Results are automatically saved to annotations/nsd/
```

#### Step 3: Post-Processing

After annotation, use the tools to clean and organize:

```python
from image_annotation.tools import remove_model, reorder_annotations

# Remove underperforming models
remove_model("annotations/nsd/", "poor_model")

# Reorder for consistency
desired_order = ["best_model", "second_best", "third"]
reorder_annotations("annotations/nsd/", desired_order)

# Export for analysis
export_to_csv("annotations/nsd/", "nsd_annotations.csv")
```

### Annotation Format

Each annotation file follows this structure:

```json
{
  "image": {
    "path": "images/original/shared0001_nsd02951.png",
    "name": "shared0001_nsd02951.png",
    "id": "shared0001_nsd02951"
  },
  "timestamp": "2025-08-30T15:42:51.595095+00:00",
  "annotations": [
    {
      "model": "gemma3:4b",
      "temperature": 0.3,
      "prompts": {
        "general_description": {
          "prompt_text": "...",
          "response": "...",
          "response_format": "text",
          "token_metrics": {
            "input_tokens": 313,
            "output_tokens": 196,
            "total_tokens": 509
          },
          "performance_metrics": {
            "total_duration_ms": 4488.649302,
            "tokens_per_second": 161.28
          }
        }
      }
    }
  ]
}
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