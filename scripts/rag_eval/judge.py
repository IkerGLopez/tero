"""Judge LLM adapter for the rag_eval runner: pricing, cost tracking and client factory."""

import config  # Bootstrap: repo .env + sys.path
import os
import sys

from openai import AsyncOpenAI
from ragas.llms import llm_factory

# ------------------------------------------------------------------
# Cost reporting
# ------------------------------------------------------------------

try:
    EMBEDDING_COST_PER_1K_TOKENS = float(os.environ.get("EMBEDDING_COST_PER_1K_TOKENS", "0.00002"))
except (ValueError, TypeError):
    sys.exit(
        f"ERROR: EMBEDDING_COST_PER_1K_TOKENS must be a number, "
        f"got: {os.environ.get('EMBEDDING_COST_PER_1K_TOKENS', '')!r}"
    )

try:
    JUDGE_COST_PER_1K_PROMPT_TOKENS = float(
        os.environ.get("JUDGE_COST_PER_1K_PROMPT_TOKENS", "0.00015")
    )
except (ValueError, TypeError):
    sys.exit(
        f"ERROR: JUDGE_COST_PER_1K_PROMPT_TOKENS must be a number, "
        f"got: {os.environ.get('JUDGE_COST_PER_1K_PROMPT_TOKENS', '')!r}"
    )

try:
    JUDGE_COST_PER_1K_COMPLETION_TOKENS = float(
        os.environ.get("JUDGE_COST_PER_1K_COMPLETION_TOKENS", "0.00060")
    )
except (ValueError, TypeError):
    sys.exit(
        f"ERROR: JUDGE_COST_PER_1K_COMPLETION_TOKENS must be a number, "
        f"got: {os.environ.get('JUDGE_COST_PER_1K_COMPLETION_TOKENS', '')!r}"
    )

# Per-model judge pricing (USD per 1K tokens), derived from public per-1M pricing:
#   gemini-3.5-flash : $1.50 / $9.00   per 1M  ->  $0.0015 / $0.0090 per 1K
#   gpt-4o           : $2.50 / $10.00  per 1M  ->  $0.0025 / $0.0100 per 1K
#   gpt-4o-mini      : $0.150 / $0.600 per 1M  ->  $0.00015 / $0.00060 per 1K
# Keys are matched case-insensitively against the --judge-model argument.
JUDGE_PRICING_TABLE = {
    "gemini-3.5-flash": (0.0015, 0.0090),
    "gemini-2.5-flash": (0.00030, 0.00250),
    "gpt-4o": (0.0025, 0.0100),
    "gpt-6-luna": (0.0001, 0.0005),
    "claude-sonnet-4": (0.00300, 0.01500),
    "claude-sonnet-4-5": (0.00300, 0.01500),
    "claude-sonnet-4-6": (0.00300, 0.01500),
    "claude-sonnet-5": (0.00200, 0.01000),
    "claude-haiku-4-5": (0.00100, 0.00500),
}

# Bedrock inference profile IDs for supported Claude judge models.
# Keys match --judge-model values; values are AWS Bedrock inference profile IDs.
# Prefix depends on your AWS region — currently set for eu-west-1 (eu.*).
# Run `aws bedrock list-inference-profiles` to find your IDs.
BEDROCK_JUDGE_MODELS: dict[str, str] = {
    "claude-sonnet-4":   "eu.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-haiku-4-5":  "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-6": "eu.anthropic.claude-sonnet-4-6",
    "claude-sonnet-5":   "eu.anthropic.claude-sonnet-5",
}

# Newer Anthropic models (adaptive thinking on by default, Sonnet 5+) deprecate
# the temperature/top_p sampling parameters — Bedrock rejects requests that
# include them. Matched as substrings against the requested model id.
NO_SAMPLING_PARAMS_MODELS: tuple[str, ...] = (
    "claude-sonnet-5",
)

def _resolve_judge_pricing(model_id: str) -> tuple:
    """Resolve per-1K-token pricing for a judge model.

    Precedence (highest wins):
      1. Explicit env vars JUDGE_COST_PER_1K_PROMPT_TOKENS /
         JUDGE_COST_PER_1K_COMPLETION_TOKENS (override everything).
      2. JUDGE_PRICING_TABLE lookup by model_id (case-insensitive).
      3. Module-level fallback constants (loaded from env with their own
         defaults; used for unknown models not in the table).

    Returns (prompt_cost_per_1k, completion_cost_per_1k) as floats.
    """
    env_prompt = os.environ.get("JUDGE_COST_PER_1K_PROMPT_TOKENS")
    env_completion = os.environ.get("JUDGE_COST_PER_1K_COMPLETION_TOKENS")
    if env_prompt is not None and env_completion is not None:
        return float(env_prompt), float(env_completion)
    entry = JUDGE_PRICING_TABLE.get(model_id.lower())
    if entry is not None:
        return entry
    return JUDGE_COST_PER_1K_PROMPT_TOKENS, JUDGE_COST_PER_1K_COMPLETION_TOKENS


class JudgeCostTracker:
    """Token-counting wrapper around the RAGAS judge LLM's API client.

    Intercepts ``chat.completions.create`` (OpenAI) or ``messages.create``
    (Anthropic Bedrock) to accumulate real ``usage`` token counts from every
    API response.  Exposes ``cost_summary()`` to print a per-run USD cost
    report formatted consistently with the existing ``cost_report()`` output.

    Pricing is resolved per model via ``JUDGE_PRICING_TABLE`` (see
    ``_resolve_judge_pricing``). Explicit ``JUDGE_COST_PER_1K_PROMPT_TOKENS`` /
    ``JUDGE_COST_PER_1K_COMPLETION_TOKENS`` env vars override the table; unknown
    models fall back to the module-level default constants.
    """

    def __init__(self, client, model_label: str = "Gemini 3.5 Flash",
                 prompt_cost_per_1k: float | None = None,
                 completion_cost_per_1k: float | None = None,
                 provider: str = "openai"):
        """Wrap *client* for token counting.

        The constructor monkey-patches the appropriate create method so
        every subsequent call is intercepted.

        *provider* ``"openai"`` (default) patches ``client.chat.completions.create``
        and reads ``usage.prompt_tokens`` / ``usage.completion_tokens``.
        ``"anthropic"`` patches ``client.messages.create`` and reads
        ``usage.input_tokens`` / ``usage.output_tokens``.
        """

        if provider == "anthropic":
            # Guard against double-wrapping on messages namespace
            if getattr(client.messages, "_tracked_by_judge_cost", None) is True:
                # Already tracked — expose minimal attributes for access safety
                self._client = client
                self._model_label = model_label
                self.prompt_cost_per_1k = 0
                self.completion_cost_per_1k = 0
                self.prompt_tokens: int = 0
                self.completion_tokens: int = 0
                return
            self._client = client
            self._model_label = model_label
            self.prompt_cost_per_1k = (
                prompt_cost_per_1k if prompt_cost_per_1k is not None
                else JUDGE_COST_PER_1K_PROMPT_TOKENS
            )
            self.completion_cost_per_1k = (
                completion_cost_per_1k if completion_cost_per_1k is not None
                else JUDGE_COST_PER_1K_COMPLETION_TOKENS
            )
            self.prompt_tokens: int = 0
            self.completion_tokens: int = 0
            self._original_create = client.messages.create
            self._wrap_anthropic()
        else:
            # OpenAI (default, backward-compatible)
            if getattr(client.chat.completions, "_tracked_by_judge_cost", None) is True:
                # Already tracked — expose minimal attributes for access safety
                self._client = client
                self._model_label = model_label
                self.prompt_cost_per_1k = 0
                self.completion_cost_per_1k = 0
                self.prompt_tokens: int = 0
                self.completion_tokens: int = 0
                return
            self._client = client
            self._model_label = model_label
            self.prompt_cost_per_1k = (
                prompt_cost_per_1k if prompt_cost_per_1k is not None
                else JUDGE_COST_PER_1K_PROMPT_TOKENS
            )
            self.completion_cost_per_1k = (
                completion_cost_per_1k if completion_cost_per_1k is not None
                else JUDGE_COST_PER_1K_COMPLETION_TOKENS
            )
            self.prompt_tokens: int = 0
            self.completion_tokens: int = 0
            self._original_create = client.chat.completions.create
            self._wrap_openai()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap_openai(self) -> None:
        """Replace ``client.chat.completions.create`` with a tracked version."""
        original = self._original_create
        # NOTE: deliberately creates reference cycle (self → _client → create →
        # closure → tracker → self). Safe for CLI scripts run once per process.
        tracker = self

        async def _tracked_create(*args, **kwargs):
            response = await original(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                tracker.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                tracker.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            return response

        self._client.chat.completions.create = _tracked_create
        self._client.chat.completions._tracked_by_judge_cost = True

    def _wrap_anthropic(self) -> None:
        """Replace ``client.messages.create`` with a tracked version
        that reads ``usage.input_tokens`` / ``usage.output_tokens``.

        Also sanitises kwargs: drops ``top_p`` and defaults ``temperature``
        to 0 (deterministic) because Bedrock Claude rejects having both set.
        Adaptive-thinking models (Sonnet 5+) reject ``temperature``/``top_p``
        entirely, so both are stripped for them.
        """
        original = self._original_create
        tracker = self

        async def _tracked_messages_create(*args, **kwargs):
            model = str(kwargs.get("model", ""))
            if any(tag in model for tag in NO_SAMPLING_PARAMS_MODELS):
                # Adaptive-thinking models (Sonnet 5+) deprecate sampling
                # params entirely; Bedrock rejects requests that include them.
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
            else:
                # Bedrock Claude rejects temperature + top_p together.
                # RAGAS/instructor may inject both; strip top_p and default
                # temperature to 0 for deterministic judge output.
                kwargs.pop("top_p", None)
                if "temperature" not in kwargs:
                    kwargs["temperature"] = 0
            response = await original(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                tracker.prompt_tokens += getattr(usage, "input_tokens", 0) or 0
                tracker.completion_tokens += getattr(usage, "output_tokens", 0) or 0
            return response

        self._client.messages.create = _tracked_messages_create
        self._client.messages._tracked_by_judge_cost = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cost_summary(self) -> None:
        """Print a per-run judge-LLM cost report to stdout.

        Format mirrors ``cost_report()``:
            === Judge Cost Report ===
            Judge LLM            : Gemini 3.5 Flash
            Prompt tokens        : N,NNN
            Completion tokens    : N,NNN
            Prompt cost/1K       : $X.XXXXXX
            Completion cost/1K   : $X.XXXXXX
            Total judge USD      : $X.XXXXXX
        """
        prompt_cost = (self.prompt_tokens / 1000) * self.prompt_cost_per_1k
        completion_cost = (self.completion_tokens / 1000) * self.completion_cost_per_1k
        total_cost = prompt_cost + completion_cost

        print()
        print("=== Judge Cost Report ===")
        print(f"Judge LLM            : {self._model_label}")
        print(f"Prompt tokens        : {self.prompt_tokens:,}")
        print(f"Completion tokens    : {self.completion_tokens:,}")
        print(f"Prompt cost/1K       : ${self.prompt_cost_per_1k:.6f}")
        print(f"Completion cost/1K   : ${self.completion_cost_per_1k:.6f}")
        print(f"Total judge USD      : ${total_cost:.6f}")


def cost_report(embedding_tokens: int, model: str = "text-embedding-3-small") -> None:
    """Print embedding token count and estimated USD cost.

    Uses cl100k_base encoding for token estimation (text-embedding-3-small/ada-002).
    Cost per 1K tokens is configurable via EMBEDDING_COST_PER_1K_TOKENS env var
    (default: $0.00002 — text-embedding-3-small at $0.02/1M tokens).
    """
    cost = (embedding_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
    print(f"\n=== Cost Report ===")
    print(f"Embedding model      : {model}")
    print(f"Embedding tokens     : {embedding_tokens:,}")
    print(f"Cost per 1K tokens   : ${EMBEDDING_COST_PER_1K_TOKENS:.6f}")
    print(f"Total estimated USD  : ${cost:.6f}")


# ------------------------------------------------------------------
# Judge LLM factory
# ------------------------------------------------------------------

def _build_judge_client(model_id: str) -> tuple:
    """Build API client, RAGAS judge LLM, and cost tracker.

    Detects vendor from model name prefix:
      - gpt*, o1*, o3*, o4* → OpenAI (OPENAI_API_KEY)
      - gemini*  → Google (GOOGLE_API_KEY)
      - claude*  → Anthropic Bedrock (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)

    Returns (client, judge_llm, cost_tracker, model_label).
    Exits with error on unknown vendor/model or missing credentials.
    """
    model_lower = model_id.lower()

    if model_lower.startswith("claude"):
        # Resolve friendly model name to Bedrock inference profile ID
        inference_profile = BEDROCK_JUDGE_MODELS.get(model_lower)
        if not inference_profile:
            valid = list(BEDROCK_JUDGE_MODELS)
            print(
                f"ERROR: Unknown Claude model '{model_id}'. "
                f"Valid: {valid}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate AWS credentials
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
            if not os.environ.get(var):
                print(
                    f"ERROR: {var} env var is required for Bedrock judge models.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # Lazy import — anthropic SDK only loaded for Bedrock judge models
        from anthropic import AsyncAnthropicBedrock

        client = AsyncAnthropicBedrock(
            aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_region=os.environ["AWS_REGION"],
            **(dict(aws_session_token=os.environ["AWS_SESSION_TOKEN"]) if os.environ.get("AWS_SESSION_TOKEN") else {}),
        )

        _prompt_rate, _completion_rate = _resolve_judge_pricing(model_id)
        cost_tracker = JudgeCostTracker(
            client,
            provider="anthropic",
            model_label=model_id,
            prompt_cost_per_1k=_prompt_rate,
            completion_cost_per_1k=_completion_rate,
        )
        judge_llm = llm_factory(
            inference_profile,
            provider="anthropic",
            client=client,
            max_tokens=16384,
        )

        return client, judge_llm, cost_tracker, model_id

    if model_lower.startswith(("gpt", "o1", "o3", "o4")):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("ERROR: OPENAI_API_KEY env var is required for OpenAI judge models.", file=sys.stderr)
            sys.exit(1)
        base_url = "https://api.openai.com/v1/"
        model_label = model_id
    elif model_lower.startswith("gemini"):
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            print("ERROR: GOOGLE_API_KEY env var is required for Gemini judge models.", file=sys.stderr)
            sys.exit(1)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model_label = model_id
    else:
        print(
            f"ERROR: Unknown judge model vendor for '{model_id}'. "
            f"Expected prefix: claude*, gpt*, o1*, o3*, o4*, or gemini*.",
            file=sys.stderr,
        )
        sys.exit(1)

    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    _prompt_rate, _completion_rate = _resolve_judge_pricing(model_id)
    cost_tracker = JudgeCostTracker(
        openai_client,
        model_label=model_label,
        prompt_cost_per_1k=_prompt_rate,
        completion_cost_per_1k=_completion_rate,
    )
    judge_llm = llm_factory(model_id, client=openai_client, max_tokens=16384)

    return openai_client, judge_llm, cost_tracker, model_label
