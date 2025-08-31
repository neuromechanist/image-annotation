# The Annotation Garden Project

🌐 **[View Live Dashboard](https://neuromechanist.github.io/image-annotation)**

A VLM-based image annotation system for Natural Scene Dataset (NSD) using multiple Vision-Language Models (VLMs).

## Key Features

- **Multi-model support**: OLLAMA, OpenAI GPT-4V, Anthropic Claude
- **Batch processing**: Handle 25k+ annotations with real-time progress
- **Web dashboard**: Interactive visualization and analysis interface
- **Annotation tools**: Reorder, filter, export, and manipulate annotations
- **Research-ready**: Structured JSON output with comprehensive metrics

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OLLAMA (for local models)
- API keys for OpenAI/Anthropic (optional)

### Installation

```bash
# Clone and setup
git clone https://github.com/neuromechanist/hed-image-annotation.git
cd hed-image-annotation

# Python environment
conda activate torch-312  # or create: conda create -n torch-312 python=3.12
pip install -e .

# Frontend
cd frontend && npm install
```

### Quick Usage

```bash
# Start OLLAMA (for local models)
ollama serve

# Test VLM service
python -m image_annotation.services.vlm_service

# Run frontend dashboard
cd frontend && npm run dev
# Visit http://localhost:3000

# Configuration
cp config/config.example.json config/config.json
# Edit config.json with API keys and NSD image paths
```

## Architecture

- **Backend**: FastAPI with OLLAMA/OpenAI/Anthropic integration
- **Frontend**: Next.js dashboard with real-time progress tracking  
- **Storage**: JSON files with database support for large datasets
- **Processing**: Stateless VLM calls with comprehensive metrics

## Annotation Tools

Powerful CLI tools for post-processing annotations:

```python
from image_annotation.utils import reorder_annotations, remove_model, export_to_csv

# Reorder model annotations
reorder_annotations("annotations/", ["best_model", "second_best"])

# Remove underperforming models
remove_model("annotations/", "poor_model")

# Export for analysis
export_to_csv("annotations/", "results.csv", include_metrics=True)
```

## Programmatic Usage

```python
from image_annotation.services.vlm_service import VLMService, VLMPrompt

# Initialize and process
service = VLMService(model="gemma3:4b")
results = service.process_batch(
    image_paths=["path/to/image.jpg"],
    prompts=[VLMPrompt(id="describe", text="Describe this image")],
    models=["gemma3:4b", "llava:latest"]
)

# Results include comprehensive metrics
for result in results:
    print(f"Tokens: {result.token_metrics.total_tokens}")
    print(f"Speed: {result.performance_metrics.tokens_per_second}/sec")
```

## Development

```bash
# Test with real data (no mocks)
pytest tests/ --cov

# Format code
ruff check --fix . && ruff format .
```


## NSD Research Usage

1. **Download NSD images** to `/path/to/NSD_stimuli/shared1000/`
2. **Configure models** in `config/config.json`
3. **Process in batches** using `VLMService.process_batch()`
4. **Post-process** with annotation tools
5. **Export results** to CSV for analysis

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed research workflows.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, standards, and submission process.

## License

This project is licensed under CC-BY-NC-SA 4.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Natural Scene Dataset (NSD) team
- LangChain and OLLAMA communities