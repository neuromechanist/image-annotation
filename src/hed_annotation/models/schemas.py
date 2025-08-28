"""JSON Schema definitions for HED Image Annotation system"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Enums for controlled vocabularies
class ServerType(str, Enum):
    OLLAMA = "OLLAMA"
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"


class ConnectionMode(str, Enum):
    STREAM = "STREAM"
    BATCH = "BATCH"


class ResponseType(str, Enum):
    TEXT = "TEXT"
    JSON = "JSON"
    STRUCTURED = "STRUCTURED"


# Model Configuration Schema
class ModelConfig(BaseModel):
    """Individual model configuration"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "llava",
                "provider": "OLLAMA",
                "version": "latest",
                "endpoint": "http://localhost:11434",
                "api_key": None,
                "parameters": {"temperature": 0.7, "max_tokens": 2000, "top_p": 0.9},
            }
        }
    )

    name: str = Field(..., description="Model name (e.g., 'llava', 'gpt-4-vision', 'claude-3')")
    provider: ServerType = Field(..., description="Provider type")
    version: str = Field(default="latest", description="Model version")
    endpoint: str | None = Field(None, description="API endpoint URL")
    api_key: str | None = Field(None, description="API key for authentication")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.7, "max_tokens": 2000},
        description="Model-specific parameters",
    )

    @field_validator("endpoint")
    def validate_endpoint(cls, v, info):
        if info.data.get("provider") == ServerType.OLLAMA and not v:
            return "http://localhost:11434"
        return v


class PromptTemplate(BaseModel):
    """Prompt template configuration"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "describe_scene",
                "display_name": "Describe Scene",
                "template": "Describe this image in detail, focusing on: {focus_areas}",
                "variables": {"focus_areas": "objects, people, setting, mood"},
                "expected_response_type": "TEXT",
                "max_length": 500,
                "validation_schema": None,
            }
        }
    )

    id: str = Field(..., description="Unique identifier for the prompt")
    display_name: str = Field(..., description="Human-readable name")
    template: str = Field(..., description="Prompt template with optional variables")
    variables: dict[str, Any] | None = Field(None, description="Default variable values")
    expected_response_type: ResponseType = Field(default=ResponseType.TEXT)
    max_length: int | None = Field(None, description="Maximum response length")
    validation_schema: dict[str, Any] | None = Field(
        None, description="JSON schema for structured responses"
    )


class ModelConfiguration(BaseModel):
    """Main configuration schema for models and prompts"""

    model_config = ConfigDict(
        protected_namespaces=(),  # Allow model_ prefix in field names
        json_schema_extra={
            "example": {
                "version": "1.0.0",
                "model_list": [],  # Empty list instead of ellipsis
                "prompts": [],  # Empty list instead of ellipsis
                "default_server": "OLLAMA",
                "connection_mode": "BATCH",
                "batch_settings": {"max_concurrent": 5, "timeout": 300},
            }
        },
    )

    version: str = Field(default="1.0.0", description="Configuration version")
    model_list: list[ModelConfig] = Field(..., description="List of configured models")
    prompts: list[PromptTemplate] = Field(..., description="List of prompt templates")
    default_server: ServerType = Field(default=ServerType.OLLAMA)
    connection_mode: ConnectionMode = Field(default=ConnectionMode.BATCH)
    batch_settings: dict[str, Any] | None = Field(
        default_factory=lambda: {"max_concurrent": 5, "timeout": 300},
        description="Batch processing settings",
    )
    stream_settings: dict[str, Any] | None = Field(
        default_factory=lambda: {"buffer_size": 1024, "flush_interval": 0.1},
        description="Stream processing settings",
    )


# Response Storage Schema
class ImageMetadata(BaseModel):
    """Image metadata"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "nsd_001",
                "filename": "image_001.jpg",
                "path": "/data/nsd_stimuli/shared1000/image_001.jpg",
                "size_bytes": 245678,
                "dimensions": {"width": 1024, "height": 768},
                "format": "JPEG",
            }
        }
    )

    id: str = Field(..., description="Unique image identifier")
    filename: str = Field(..., description="Image filename")
    path: str = Field(..., description="Full path to image file")
    size_bytes: int | None = Field(None, description="File size in bytes")
    dimensions: dict[str, int] | None = Field(None, description="Image dimensions")
    format: str | None = Field(None, description="Image format")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional image metadata"
    )


class ModelDetails(BaseModel):
    """Model execution details"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "llava",
                "provider": "OLLAMA",
                "version": "latest",
                "parameters": {"temperature": 0.7, "max_tokens": 2000},
            }
        }
    )

    name: str = Field(..., description="Model name used")
    provider: ServerType = Field(..., description="Provider used")
    version: str = Field(..., description="Model version")
    parameters: dict[str, Any] = Field(..., description="Parameters used for this run")


class PromptInfo(BaseModel):
    """Prompt execution information"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "template_id": "describe_scene",
                "display_name": "Describe Scene",
                "actual_prompt": "Describe this image in detail, focusing on: objects, people, setting, mood",
                "variables_used": {"focus_areas": "objects, people, setting, mood"},
            }
        }
    )

    template_id: str = Field(..., description="Prompt template ID")
    display_name: str = Field(..., description="Human-readable prompt name")
    actual_prompt: str = Field(..., description="Actual prompt sent to model")
    variables_used: dict[str, Any] | None = Field(None, description="Variables substituted")


class ExecutionStats(BaseModel):
    """Execution statistics"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tokens_used": {"prompt": 150, "completion": 350, "total": 500},
                "processing_time_ms": 2345,
                "temperature": 0.7,
                "timestamp": "2024-08-27T12:34:56Z",
                "success": True,
            }
        }
    )

    tokens_used: dict[str, int] | None = Field(None, description="Token usage breakdown")
    processing_time_ms: int | None = Field(None, description="Processing time in milliseconds")
    temperature: float | None = Field(None, description="Temperature parameter used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(default=True, description="Whether execution succeeded")
    error_message: str | None = Field(None, description="Error message if failed")


class AnnotationResponse(BaseModel):
    """Complete annotation response"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "annotation_001",
                "image": {},  # Empty dict instead of ellipsis
                "model": {},  # Empty dict instead of ellipsis
                "prompt": {},  # Empty dict instead of ellipsis
                "stats": {},  # Empty dict instead of ellipsis
                "answer": "The image shows a serene landscape with...",
                "answer_type": "TEXT",
                "structured_answer": None,
            }
        }
    )

    id: str = Field(..., description="Unique annotation ID")
    image: ImageMetadata = Field(..., description="Image metadata")
    model: ModelDetails = Field(..., description="Model details")
    prompt: PromptInfo = Field(..., description="Prompt information")
    stats: ExecutionStats = Field(..., description="Execution statistics")
    answer: str = Field(..., description="Raw response from model")
    answer_type: ResponseType = Field(default=ResponseType.TEXT)
    structured_answer: dict[str, Any] | None = Field(
        None, description="Parsed structured response if applicable"
    )

    @field_validator("structured_answer")
    def validate_structured_answer(cls, v, info):
        if info.data.get("answer_type") == ResponseType.JSON and v is None:
            # Try to parse JSON from answer
            import json

            try:
                return json.loads(info.data.get("answer", "{}"))
            except (json.JSONDecodeError, TypeError):
                return None
        return v


# Batch Response Schema
class BatchAnnotationResponse(BaseModel):
    """Batch annotation response container"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "batch_id": "batch_001",
                "total_images": 10,
                "total_prompts": 5,
                "total_annotations": 50,
                "annotations": [],  # Empty list instead of ellipsis
                "batch_stats": {
                    "start_time": "2024-08-27T12:00:00Z",
                    "end_time": "2024-08-27T12:30:00Z",
                    "success_count": 48,
                    "failure_count": 2,
                },
            }
        }
    )

    batch_id: str = Field(..., description="Unique batch identifier")
    total_images: int = Field(..., description="Number of images processed")
    total_prompts: int = Field(..., description="Number of prompts used")
    total_annotations: int = Field(..., description="Total annotations generated")
    annotations: list[AnnotationResponse] = Field(..., description="List of annotations")
    batch_stats: dict[str, Any] = Field(..., description="Batch-level statistics")


# Export functions for JSON Schema generation
def get_model_configuration_schema():
    """Get JSON Schema for model configuration"""
    return ModelConfiguration.model_json_schema()


def get_annotation_response_schema():
    """Get JSON Schema for annotation response"""
    return AnnotationResponse.model_json_schema()


def get_batch_response_schema():
    """Get JSON Schema for batch response"""
    return BatchAnnotationResponse.model_json_schema()


# Example instances
if __name__ == "__main__":
    import json

    # Example model configuration
    config = ModelConfiguration(
        model_list=[
            ModelConfig(
                name="llava", provider=ServerType.OLLAMA, endpoint="http://localhost:11434"
            ),
            ModelConfig(name="gpt-4-vision-preview", provider=ServerType.OPENAI, api_key="sk-..."),
        ],
        prompts=[
            PromptTemplate(
                id="describe",
                display_name="Describe Image",
                template="Describe this image in 200 words",
            ),
            PromptTemplate(
                id="keywords",
                display_name="Extract Keywords",
                template="List 10 keywords for this image as JSON",
                expected_response_type=ResponseType.JSON,
                validation_schema={"type": "array", "items": {"type": "string"}},
            ),
        ],
    )

    print("Model Configuration Example:")
    print(json.dumps(config.model_dump(), indent=2))

    print("\n\nJSON Schema for Model Configuration:")
    print(json.dumps(get_model_configuration_schema(), indent=2))
