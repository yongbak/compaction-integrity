import argparse
from typing import Any

from openai import OpenAI

from compaction_integrity.dataset.eval_loader import Message
from compaction_integrity.prompts import get_summarization_prompt
from compaction_integrity.runtime.base import ModelRuntime
from compaction_integrity.runtime.vllm_serve_runtime import VLLMServeRuntime

DEFAULT_MODEL_IP = "192.168.251.57:8000"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Option names accepted by get_summarization_prompt() (prompts.py:265)
PROMPT_TEMPLATES = {
    "anthropic": "Claude compaction docs prompt; transcript + summary instruction, output in <summary>",
    "anthropic-sc-targeted": "anthropic + explicitly preserve user session-level constraints (Appendix B.3)",
    "google adk": "Google ADK event summarizer; short free-form summary",
    "pi_mono": "openclaw/pi-mono structured checkpoint (Goal / Constraints / Progress / Next Steps)",
}
DEFAULT_PROMPT_TEMPLATE = "anthropic"


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


def compact(
    messages: list[Message],
    model_ip: str = DEFAULT_MODEL_IP,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_tokens: int = 4096,
) -> list[Message]:
    runtime = ExistingVLLMServerRuntime(
        config={"model": model_name, "base_url": _to_base_url(model_ip), "max_tokens": max_tokens}
    )
    try:
        response = runtime.generate(
            messages=get_summarization_prompt(prompt_template, messages),
            model=model_name,
        )
    finally:
        runtime.close()
    # Same shape as LLMSummarizeCompactor._build_compaction_result().messages
    return [{"role": "assistant", "content": response.text}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=list(PROMPT_TEMPLATES), default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--model_ip", default=DEFAULT_MODEL_IP)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    conversation: list[Message] = [
        {"role": "user", "content": "For the rest of this session, show me the draft before sending anything.\nHelp me write an email to Sarah."},
        {"role": "assistant", "content": "Sure. What should the email say?"},
        {"role": "user", "content": "Tell her I'll be 10 minutes late to the meeting."},
        {"role": "assistant", "content": "Here's a draft: 'Hi Sarah, I'll be about 10 minutes late to our meeting. Sorry!'"},
        {"role": "user", "content": "It sound too casual. Please make it more politely."},
    ]
    print(f"[prompt] {args.prompt}: {PROMPT_TEMPLATES[args.prompt]}")
    print(f"[model]  {args.model_name} @ {args.model_ip}")
    compacted = compact(
        conversation,
        model_ip=args.model_ip,
        model_name=args.model_name,
        prompt_template=args.prompt,
    )
    print(compacted)

# Compaction test (after `pip install -e .`; otherwise prefix with PYTHONPATH=src from the repo root):
#   python -m compaction_integrity.compaction                           # default prompt (anthropic)
#   python -m compaction_integrity.compaction --prompt pi_mono
#   python -m compaction_integrity.compaction --prompt anthropic-sc-targeted
#   python -m compaction_integrity.compaction --prompt "google adk"
#
# Use from other code:
#   from compaction_integrity.compaction import compact
#   compacted = compact(messages, prompt_template="pi_mono")
