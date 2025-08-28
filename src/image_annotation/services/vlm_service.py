"""VLM Service with comprehensive metrics using LangChain ChatOllama for image annotation."""

import base64
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class VLMPrompt(BaseModel):
    """Configuration for a VLM prompt."""

    id: str = Field(..., description="Unique identifier for the prompt")
    text: str = Field(..., description="The prompt text")
    expected_format: str = Field(
        default="text", description="Expected response format: text or json"
    )


class TokenMetrics(BaseModel):
    """Token usage metrics from the model."""

    input_tokens: int | None = Field(None, description="Number of prompt/input tokens")
    output_tokens: int | None = Field(None, description="Number of generated/output tokens")
    total_tokens: int | None = Field(None, description="Total tokens (input + output)")


class PerformanceMetrics(BaseModel):
    """Performance metrics for the inference."""

    total_duration_ms: float | None = Field(None, description="Total request time in milliseconds")
    prompt_eval_duration_ms: float | None = Field(
        None, description="Prompt evaluation time in milliseconds"
    )
    generation_duration_ms: float | None = Field(
        None, description="Response generation time in milliseconds"
    )
    load_duration_ms: float | None = Field(None, description="Model load time in milliseconds")
    tokens_per_second: float | None = Field(None, description="Generation speed in tokens/second")


class VLMResult(BaseModel):
    """Result from a VLM annotation with comprehensive metrics."""

    # Basic fields
    image_path: str
    model: str
    prompt_id: str
    prompt_text: str
    response: str  # Always stores raw string response
    response_format: str
    response_data: dict | None = None  # Parsed JSON when response_format is "json"

    # Metrics
    token_metrics: TokenMetrics | None = None
    performance_metrics: PerformanceMetrics | None = None

    # Status
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Additional metadata
    temperature: float | None = None


class VLMService:
    """VLM service with comprehensive metrics collection using LangChain ChatOllama."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
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

    def _extract_json_from_markdown(self, response: str) -> str:
        """Extract JSON from markdown code blocks if present.

        Args:
            response: Raw response that might contain markdown-wrapped JSON.

        Returns:
            Extracted JSON string or original response.
        """
        if "```json" in response:
            # Extract content between ```json and ```
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in response:
            # Extract content between ``` and ```
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return response

    def _parse_metrics(self, ai_message: AIMessage) -> tuple[TokenMetrics, PerformanceMetrics]:
        """Parse metrics from AIMessage response.

        Args:
            ai_message: The AI response message.

        Returns:
            Tuple of (TokenMetrics, PerformanceMetrics).
        """
        # Extract token metrics from usage_metadata
        token_metrics = TokenMetrics()
        if hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata:
            usage = ai_message.usage_metadata
            token_metrics.input_tokens = usage.get("input_tokens")
            token_metrics.output_tokens = usage.get("output_tokens")
            token_metrics.total_tokens = usage.get("total_tokens")

        # Extract performance metrics from response_metadata
        performance_metrics = PerformanceMetrics()
        if hasattr(ai_message, "response_metadata") and ai_message.response_metadata:
            metadata = ai_message.response_metadata

            # Convert nanoseconds to milliseconds for OLLAMA metrics
            if "total_duration" in metadata:
                performance_metrics.total_duration_ms = metadata["total_duration"] / 1_000_000

            if "prompt_eval_duration" in metadata:
                performance_metrics.prompt_eval_duration_ms = (
                    metadata["prompt_eval_duration"] / 1_000_000
                )

            if "eval_duration" in metadata:
                performance_metrics.generation_duration_ms = metadata["eval_duration"] / 1_000_000

            if "load_duration" in metadata:
                performance_metrics.load_duration_ms = metadata["load_duration"] / 1_000_000

            # Calculate tokens per second
            if (
                "eval_count" in metadata
                and "eval_duration" in metadata
                and metadata["eval_duration"] > 0
            ):
                eval_duration_seconds = metadata["eval_duration"] / 1_000_000_000
                performance_metrics.tokens_per_second = (
                    metadata["eval_count"] / eval_duration_seconds
                )

        return token_metrics, performance_metrics

    def annotate_image(
        self, image_path: str | Path, prompt: VLMPrompt, model: str | None = None
    ) -> VLMResult:
        """Annotate a single image with a single prompt, collecting comprehensive metrics.

        Args:
            image_path: Path to the image file.
            prompt: Prompt configuration.
            model: Optional model override.

        Returns:
            VLMResult with response and comprehensive metrics.
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

            # Run inference (returns AIMessage with metadata)
            ai_message = llm.invoke([message])

            # Extract response content
            response = (
                ai_message.content
                if isinstance(ai_message.content, str)
                else str(ai_message.content)
            )

            # Parse metrics
            token_metrics, performance_metrics = self._parse_metrics(ai_message)

            # Handle JSON extraction if needed
            response_data = None
            error = None

            if prompt.expected_format == "json":
                try:
                    json_str = self._extract_json_from_markdown(response)
                    response_data = json.loads(json_str)
                    # Keep the raw response for preservation, but also store parsed data
                except json.JSONDecodeError as e:
                    error = f"JSON parsing failed: {e}"
                    # Still return the result with the raw response and error

            return VLMResult(
                image_path=str(image_path),
                model=model or self.model,
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                response=response,  # Always store raw response
                response_format=prompt.expected_format,
                response_data=response_data,  # Parsed JSON when applicable
                token_metrics=token_metrics,
                performance_metrics=performance_metrics,
                temperature=self.temperature,
                error=error,
            )

        except Exception as e:
            # Still try to calculate basic timing
            total_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

            return VLMResult(
                image_path=str(image_path),
                model=model or self.model,
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                response="",
                response_format=prompt.expected_format,
                performance_metrics=PerformanceMetrics(total_duration_ms=total_time_ms),
                temperature=self.temperature,
                error=str(e),
            )

    def annotate_batch(
        self,
        image_paths: list[str | Path],
        prompts: list[VLMPrompt],
        models: list[str] | None = None,
    ) -> list[VLMResult]:
        """Annotate multiple images with multiple prompts and models.

        Args:
            image_paths: List of image paths.
            prompts: List of prompt configurations.
            models: Optional list of models to use.

        Returns:
            List of VLMResults with comprehensive metrics.
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

                    # Print quick metrics summary
                    if result.token_metrics and result.performance_metrics:
                        print(
                            f"  → Tokens: {result.token_metrics.total_tokens} | "
                            f"Speed: {result.performance_metrics.tokens_per_second:.1f} tok/s | "
                            f"Time: {result.performance_metrics.total_duration_ms:.0f}ms"
                        )

        return results


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
        results: List of annotation results with metrics.
        output_dir: Directory to save results.

    Returns:
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate aggregate metrics
    total_tokens = sum(
        r.token_metrics.total_tokens
        for r in results
        if r.token_metrics and r.token_metrics.total_tokens
    )
    avg_speed = sum(
        r.performance_metrics.tokens_per_second
        for r in results
        if r.performance_metrics and r.performance_metrics.tokens_per_second
    )
    num_with_speed = sum(
        1 for r in results if r.performance_metrics and r.performance_metrics.tokens_per_second
    )

    if num_with_speed > 0:
        avg_speed = avg_speed / num_with_speed

    # Convert results to dict format
    results_dict = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_annotations": len(results),
            "successful": sum(1 for r in results if r.error is None),
            "failed": sum(1 for r in results if r.error is not None),
            "total_tokens_used": total_tokens,
            "average_tokens_per_second": round(avg_speed, 2) if num_with_speed > 0 else None,
        },
        "annotations": [r.model_dump(mode="json") for r in results],
    }

    # Save to JSON with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"vlm_annotations_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    return output_file


def print_detailed_metrics(result: VLMResult):
    """Print detailed metrics for a single result."""
    print(f"\n{'=' * 60}")
    print(f"Prompt: {result.prompt_id}")
    print(f"Model: {result.model} (temp={result.temperature})")

    if result.error:
        print(f"❌ Error: {result.error}")
    else:
        print("✅ Success")

        # Token metrics
        if result.token_metrics:
            print("\nToken Metrics:")
            print(f"  • Input tokens: {result.token_metrics.input_tokens}")
            print(f"  • Output tokens: {result.token_metrics.output_tokens}")
            print(f"  • Total tokens: {result.token_metrics.total_tokens}")

        # Performance metrics
        if result.performance_metrics:
            print("\nPerformance Metrics:")
            if result.performance_metrics.total_duration_ms:
                print(f"  • Total time: {result.performance_metrics.total_duration_ms:.1f}ms")
            if result.performance_metrics.prompt_eval_duration_ms:
                print(
                    f"  • Prompt eval: {result.performance_metrics.prompt_eval_duration_ms:.1f}ms"
                )
            if result.performance_metrics.generation_duration_ms:
                print(f"  • Generation: {result.performance_metrics.generation_duration_ms:.1f}ms")
            if result.performance_metrics.tokens_per_second:
                print(
                    f"  • Speed: {result.performance_metrics.tokens_per_second:.1f} tokens/second"
                )

        # Response preview
        print(f"\nResponse ({result.response_format}):")
        response_preview = (
            result.response[:300] + "..." if len(result.response) > 300 else result.response
        )
        print(f"{response_preview}")

    print("-" * 60)


def main():
    """Test the VLM service with comprehensive metrics."""
    print("=" * 60)
    print("VLM Service Test with Metrics")
    print("=" * 60)

    # Configuration - use relative paths from project root
    image_dir = Path("images/downsampled")
    output_dir = Path("annotations/test")

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

    # Run annotation
    print(f"\n{'=' * 60}")
    print(f"Running {len(prompts)} prompts on {first_image.name}...")
    print(f"{'=' * 60}\n")

    results = service.annotate_batch([first_image], prompts)

    # Print detailed metrics for each result
    for result in results:
        print_detailed_metrics(result)

    # Save results
    output_file = save_results(results, output_dir)
    print(f"\n✅ Results saved to: {output_file}")

    # Summary with metrics
    successful = sum(1 for r in results if r.error is None)
    failed = sum(1 for r in results if r.error is not None)
    total_tokens = sum(
        r.token_metrics.total_tokens
        for r in results
        if r.token_metrics and r.token_metrics.total_tokens
    )

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  • Total annotations: {len(results)}")
    print(f"  • Successful: {successful}")
    print(f"  • Failed: {failed}")
    print(f"  • Total tokens used: {total_tokens}")

    # Average performance
    speeds = [
        r.performance_metrics.tokens_per_second
        for r in results
        if r.performance_metrics and r.performance_metrics.tokens_per_second
    ]
    if speeds:
        print(f"  • Average speed: {sum(speeds) / len(speeds):.1f} tokens/second")

    print("=" * 60)


if __name__ == "__main__":
    main()
