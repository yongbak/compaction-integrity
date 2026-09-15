import argparse

from compaction_integrity.compactors.remote_llm_summarize import RemoteLLMSummarizeCompactor
from compaction_integrity.prompts import PromptMessage

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


def compact(
    conversation: list[PromptMessage],
    model_ip: str,
    model_name: str,
    prompt_template: str,
    max_tokens: int = 4096,
) -> str:
    compactor = RemoteLLMSummarizeCompactor(
        model=model_name,
        model_ip=model_ip,
        prompt_template=prompt_template,
        runtime_kwargs={"max_tokens": max_tokens},
    )
    try:
        return compactor.summarize(conversation)
    finally:
        compactor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=list(PROMPT_TEMPLATES), default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--model_ip", default=DEFAULT_MODEL_IP)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    conversation: list[PromptMessage] = [
        {"role": "user", "content": "For the rest of this session, show me the draft before sending anything.\nHelp me write an email to Sarah."},
        {"role": "assistant", "content": "Sure. What should the email say?"},
        {"role": "user", "content": "Tell her I'll be 10 minutes late to the meeting."},
        {"role": "assistant", "content": "Here's a draft: 'Hi Sarah, I'll be about 10 minutes late to our meeting. Sorry!'"},
        {"role": "user", "content": "It sound too casual. Please make it more politely."},
    ]
    print(f"[prompt] {args.prompt}: {PROMPT_TEMPLATES[args.prompt]}")
    print(f"[model]  {args.model_name} @ {args.model_ip}")
    summary = compact(
        conversation,
        model_ip=args.model_ip,
        model_name=args.model_name,
        prompt_template=args.prompt,
    )
    print(summary)

# Compaction test (after `pip install -e .`; otherwise prefix with PYTHONPATH=src from the repo root):
#   python -m compaction_integrity.compaction                           # default prompt (anthropic)
#   python -m compaction_integrity.compaction --prompt pi_mono
#   python -m compaction_integrity.compaction --prompt anthropic-sc-targeted
#   python -m compaction_integrity.compaction --prompt "google adk"
#
# Use from other code:
#   from compaction_integrity.compaction import compact
#   summary = compact(conversation, model_ip="IP:PORT", model_name="...", prompt_template="pi_mono")  # -> str
