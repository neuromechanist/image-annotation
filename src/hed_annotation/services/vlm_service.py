"""VLM Service using LangChain ChatOllama for image annotation."""

import base64
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class VLMPrompt(BaseModel):
    """Configuration for a VLM prompt."""

    id: str = Field(..., description="Unique identifier for the prompt")
    text: str = Field(..., description="The prompt text")
    expected_format: str = Field(
        default="text", description="Expected response format: text or json"
    )


class VLMResult(BaseModel):
    """Result from a VLM annotation."""

    image_path: str
    model: str
    prompt_id: str
    prompt_text: str
    response: str
    response_format: str
    processing_time_ms: int | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VLMService:
    """Service for VLM inference using LangChain ChatOllama."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout: int = 60,
    ):
        """Initialize the VLM service.

        Args:
            model: OLLAMA model name.
            base_url: OLLAMA API base URL.
            temperature: Generation temperature.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self._llm_cache = {}

    def _get_llm(self, model: str | None = None) -> ChatOllama:
        """Get or create a ChatOllama instance.

        Args:
            model: Optional model override.

        Returns:
            ChatOllama instance.
        """
        model_name = model or self.model

        # Cache LLM instances per model
        if model_name not in self._llm_cache:
            self._llm_cache[model_name] = ChatOllama(
                model=model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=self.timeout,
            )

        return self._llm_cache[model_name]

    def _encode_image(self, image_path: str | Path) -> str:
        """Encode an image file to base64.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64 encoded string.

        Raises:
            FileNotFoundError: If image file doesn't exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _create_message(self, prompt_text: str, image_b64: str) -> HumanMessage:
        """Create a HumanMessage with image and text content.

        Args:
            prompt_text: The prompt text.
            image_b64: Base64 encoded image.

        Returns:
            HumanMessage with multimodal content.
        """
        return HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                },
                {"type": "text", "text": prompt_text},
            ]
        )

    def annotate_image(
        self, image_path: str | Path, prompt: VLMPrompt, model: str | None = None
    ) -> VLMResult:
        """Annotate a single image with a single prompt.

        Args:
            image_path: Path to the image file.
            prompt: Prompt configuration.
            model: Optional model override.

        Returns:
            VLMResult with the response or error.
        """
        image_path = Path(image_path)
        start_time = datetime.now(UTC)

        try:
            # Encode image
            image_b64 = self._encode_image(image_path)

            # Get LLM instance
            llm = self._get_llm(model)

            # Create message
            message = self._create_message(prompt.text, image_b64)

            # Create chain with output parser
            chain = llm | StrOutputParser()

            # Run inference (fresh context per prompt)
            response = chain.invoke([message])

            # Calculate processing time
            processing_time_ms = int(
                (datetime.now(UTC) - start_time).total_seconds() * 1000
            )

            # Validate JSON if expected
            if prompt.expected_format == "json":
                try:
                    # Extract JSON from markdown code blocks if present
                    json_str = response
                    if "```json" in response:
                        # Extract content between ```json and ```
                        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                        if match:
                            json_str = match.group(1)
                    elif "```" in response:
                        # Extract content between ``` and ```
                        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
                        if match:
                            json_str = match.group(1)

                    # Try to parse the extracted JSON
                    parsed = json.loads(json_str)
                    # Update response with clean JSON
                    response = json.dumps(parsed, indent=2)
                except json.JSONDecodeError as e:
                    return VLMResult(
                        image_path=str(image_path),
                        model=model or self.model,
                        prompt_id=prompt.id,
                        prompt_text=prompt.text,
                        response=response,
                        response_format=prompt.expected_format,
                        processing_time_ms=processing_time_ms,
                        error=f"JSON parsing failed: {e}",
                    )

            return VLMResult(
                image_path=str(image_path),
                model=model or self.model,
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                response=response,
                response_format=prompt.expected_format,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            processing_time_ms = int(
                (datetime.now(UTC) - start_time).total_seconds() * 1000
            )
            return VLMResult(
                image_path=str(image_path),
                model=model or self.model,
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                response="",
                response_format=prompt.expected_format,
                processing_time_ms=processing_time_ms,
                error=str(e),
            )

    def annotate_batch(
        self,
        image_paths: list[str | Path],
        prompts: list[VLMPrompt],
        models: list[str] | None = None,
    ) -> list[VLMResult]:
        """Annotate multiple images with multiple prompts and models.

        This ensures fresh context for each prompt (stateless processing).

        Args:
            image_paths: List of image paths.
            prompts: List of prompt configurations.
            models: Optional list of models to use.

        Returns:
            List of VLMResults for all combinations.
        """
        results = []
        models = models or [self.model]

        total_combinations = len(image_paths) * len(models) * len(prompts)
        current = 0

        for image_path in image_paths:
            for model in models:
                for prompt in prompts:
                    current += 1
                    print(
                        f"Processing {current}/{total_combinations}: {Path(image_path).name} - {model} - {prompt.id}"
                    )

                    result = self.annotate_image(image_path, prompt, model)
                    results.append(result)

        return results

    def process_directory(
        self,
        directory: str | Path,
        prompts: list[VLMPrompt],
        models: list[str] | None = None,
        pattern: str = "*.jpg",
        limit: int | None = None,
    ) -> list[VLMResult]:
        """Process images in a directory.

        Args:
            directory: Directory containing images.
            prompts: List of prompt configurations.
            models: Optional list of models to use.
            pattern: Glob pattern for image files.
            limit: Optional limit on number of images to process.

        Returns:
            List of VLMResults.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find matching images
        image_paths = sorted(directory.glob(pattern))
        if not image_paths:
            raise ValueError(f"No images found matching pattern {pattern} in {directory}")

        # Apply limit if specified
        if limit:
            image_paths = image_paths[:limit]

        print(f"Found {len(image_paths)} images to process")
        return self.annotate_batch(image_paths, prompts, models)


def create_test_prompts() -> list[VLMPrompt]:
    """Create the test prompts as specified."""
    return [
        VLMPrompt(
            id="describe_100",
            text="Describe this image in 100 words",
            expected_format="text",
        ),
        VLMPrompt(
            id="list_objects",
            text=(
                "List items and objects and people that you see in this image in a JSON format "
                "under human, animal, natural, and man-made main keys, "
                "with each item as subkey and the count as the value."
            ),
            expected_format="json",
        ),
    ]


def save_results(results: list[VLMResult], output_dir: str | Path) -> Path:
    """Save annotation results to JSON file.

    Args:
        results: List of annotation results.
        output_dir: Directory to save results.

    Returns:
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to dict format
    results_dict = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_annotations": len(results),
            "successful": sum(1 for r in results if r.error is None),
            "failed": sum(1 for r in results if r.error is not None),
        },
        "annotations": [r.model_dump(mode="json") for r in results],
    }

    # Save to JSON with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"vlm_annotations_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    return output_file


def main():
    """Test the VLM service with specified prompts and first image."""
    print("=" * 50)
    print("VLM Service Test")
    print("=" * 50)

    # Configuration
    image_dir = Path("/Users/yahya/Documents/git/hed-image-annotation/images/downsampled")
    output_dir = Path("/Users/yahya/Documents/git/hed-image-annotation/annotations/test")

    # Get first image
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"No images found in {image_dir}")
        return

    first_image = images[0]
    print(f"\nTest image: {first_image.name}")
    print("Model: gemma3:4b")
    print("OLLAMA endpoint: http://localhost:11434")

    # Create service
    service = VLMService(model="gemma3:4b", temperature=0.7)

    # Create test prompts
    prompts = create_test_prompts()
    print("\nPrompts to test:")
    for prompt in prompts:
        print(f"  - {prompt.id}: {prompt.text[:50]}...")

    # Run annotation on first image only
    print(f"\n{'=' * 50}")
    print(f"Running {len(prompts)} prompts on {first_image.name}...")
    print(f"{'=' * 50}\n")

    results = service.annotate_batch([first_image], prompts)

    # Print results
    for result in results:
        print(f"\nPrompt: {result.prompt_id}")
        print(f"Processing time: {result.processing_time_ms}ms")

        if result.error:
            print(f"❌ Error: {result.error}")
        else:
            print("✅ Success")
            print(f"Response ({result.response_format}):")
            # Truncate long responses for display
            response_preview = (
                result.response[:500] + "..." if len(result.response) > 500 else result.response
            )
            print(f"{response_preview}")
        print("-" * 50)

    # Save results
    output_file = save_results(results, output_dir)
    print(f"\n✅ Results saved to: {output_file}")

    # Summary
    successful = sum(1 for r in results if r.error is None)
    failed = sum(1 for r in results if r.error is not None)
    print("\nSummary:")
    print(f"  Total annotations: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
