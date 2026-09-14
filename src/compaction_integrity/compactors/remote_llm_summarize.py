from typing import Any

from openai import OpenAI

from compaction_integrity.compactors.base import CompactionResult, Compactor
from compaction_integrity.prompts import get_summarization_prompt
from compaction_integrity.runtime.base import ModelRuntime
from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime
from compaction_integrity.tokenization import count_tokens_messages


class ExistingVLLMServerRuntime(VLLMServeRuntime):
    """Reuses VLLMServeRuntime.generate() against an already-running server instead of spawning one."""

    def __init__(self, config: dict[str, Any]):
        ModelRuntime.__init__(self, config)
        self.model = str(config["model"])
        self.client = OpenAI(base_url=str(config["base_url"]), api_key="EMPTY")

    def close(self) -> None:
        self.client.close()


def _to_base_url(model_ip: str) -> str:
    url = model_ip if model_ip.startswith(("http://", "https://")) else f"http://{model_ip}"
    url = url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


# Same as LLMSummarizeCompactor, but talks to an already-running OpenAI-compatible
# server; kept separate because importing llm_summarize requires vllm to be installed.
class RemoteLLMSummarizeCompactor(Compactor):
    def __init__(
        self,
        model: str,
        model_ip: str,
        prompt_template: str,
        runtime_kwargs: dict[str, Any] | None = None,
        extra_instruction: str | None = None,
    ):
        self.model = model
        self.base_url = _to_base_url(model_ip)
        self.prompt_template = prompt_template
        self.extra_instruction = extra_instruction
        self.runtime = ExistingVLLMServerRuntime(
            config={"model": model, "base_url": self.base_url, **(runtime_kwargs or {})}
        )

    def name(self) -> str:
        return f"remote_llm_summarize_{self.model}_{self.prompt_template}"

    def _build_prompt(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        prompt = get_summarization_prompt(self.prompt_template, messages)
        if self.extra_instruction:
            # Every template ends with its summarization instruction, so the extra text lands next to it.
            last = prompt[-1]
            prompt[-1] = {**last, "content": f"{last['content']}\n\n{self.extra_instruction}"}
        return prompt

    def summarize(self, messages: list[dict[str, Any]]) -> str:
        response = self.runtime.generate(messages=self._build_prompt(messages), model=self.model)
        return response.text

    def _build_compaction_result(
        self,
        source_messages: list[dict[str, Any]],
        summary_text: str,
        target_tokens: int | None,
    ) -> CompactionResult:
        tokens_before = count_tokens_messages(source_messages)
        compacted_messages = [{"role": "assistant", "content": summary_text}]
        tokens_after = count_tokens_messages(compacted_messages)
        compression_ratio = (tokens_after / tokens_before) if tokens_before else 1.0
        return CompactionResult(
            messages=compacted_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compression_ratio=compression_ratio,
            notes={
                "base_url": self.base_url,
                "model": self.model,
                "prompt_template": self.prompt_template,
                "target_tokens": target_tokens,
            },
        )

    def compact(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int = None,
    ) -> CompactionResult:
        return self._build_compaction_result(
            source_messages=messages,
            summary_text=self.summarize(messages),
            target_tokens=target_tokens,
        )

    def compact_batch(
        self,
        batch_messages: list[list[dict[str, Any]]],
        target_tokens: int = None,
    ) -> list[CompactionResult]:
        if not batch_messages:
            return []

        responses = self.runtime.batch_generate(
            batch_messages=[self._build_prompt(messages) for messages in batch_messages],
            model=self.model,
        )
        return [
            self._build_compaction_result(
                source_messages=messages,
                summary_text=response.text,
                target_tokens=target_tokens,
            )
            for messages, response in zip(batch_messages, responses)
        ]

    def close(self) -> None:
        self.runtime.close()
