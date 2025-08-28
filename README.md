# HED Image Annotation System

A VLM-based image annotation system for Natural Scene Dataset (NSD) to Hierarchical Event Descriptors using multiple language models.

## Overview

This system provides a comprehensive solution for annotating images using various Vision-Language Models (VLMs) including OLLAMA, OpenAI, and Anthropic. It processes images from the Natural Scene Dataset and generates structured annotations that can be used for research and analysis.

## Features

- Multi-model support (OLLAMA, OpenAI GPT-4V, Anthropic Claude)
- Batch processing with real-time progress tracking
- Structured JSON output with schema validation
- Web-based interface for visualization and interaction
- Database storage for efficient querying of 25k+ annotations
- Export functionality for research analysis

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OLLAMA (for local models)
- API keys for OpenAI/Anthropic (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hed-image-annotation.git
cd hed-image-annotation
```

2. Create and activate a conda environment:
```bash
conda create -n hed-annotation python=3.11
conda activate hed-annotation
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

### Configuration

1. Copy the example configuration:
```bash
cp config/config.example.json config/config.json
```

2. Edit `config/config.json` to add your API keys and model preferences

3. Set up NSD images path in configuration

### Usage

1. Start the backend server:
```bash
uvicorn src.hed_annotation.api.main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Open your browser to http://localhost:3000

## Project Structure

```
hed-image-annotation/
├── src/
│   └── hed_annotation/
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