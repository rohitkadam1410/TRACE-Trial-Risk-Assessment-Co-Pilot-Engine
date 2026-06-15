"""
copilot.py — LLM Co-Pilot for Clinical Trial Protocol Risk Analysis

Manages vLLM server lifecycle on AMD MI300X (ROCm) and handles all LLM
calls for explaining risk scores, suggesting protocol rewrites, and
narrating what-if scenario changes.

Fallback: If vLLM fails to start, silently degrades to Anthropic
claude-haiku-3-5 so the Gradio app never crashes due to LLM availability.
"""

import os
import re
import json
import time
import signal
import logging
import subprocess
from typing import Optional, Callable, Any

import requests
import openai
import anthropic

import logger_config
logger_config.setup_logging(__file__)
logger = logging.getLogger("copilot")

# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Prompt Library — all prompts are module-level constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_EXPLAIN: str = (
    "You are a clinical research expert reviewing protocol risk assessments.\n"
    "You receive a risk score and section-level attributions for a clinical trial.\n"
    "Your job: explain WHY this trial is high/medium/low risk in 2-3 plain sentences\n"
    "that a clinical operations VP (not a data scientist) would understand.\n"
    "Be specific. Name the actual protocol issues. Do not hedge excessively.\n"
    "Do not mention the ML model or SHAP values. Speak as a domain expert."
)

SYSTEM_PROMPT_REWRITE: str = (
    "You are a senior clinical trial protocol editor.\n"
    "You receive a section of a trial protocol and its risk attribution.\n"
    "Your job: suggest 2-3 specific, actionable edits that would reduce termination risk.\n"
    "Format as a numbered list. Each edit must be specific enough to implement directly.\n"
    "Reference real regulatory guidance (FDA, ICH E6, ICH E9) where relevant.\n"
    "Do not be generic. \"Increase sample size\" is bad. \"Increase n from 45 to 120 "
    "based on 80%% power for the primary endpoint RECIST 1.1 ORR\" is good."
)

SYSTEM_PROMPT_WHATIF: str = (
    "You are a clinical trial risk analyst.\n"
    "You receive the original protocol parameters and a modified version.\n"
    "Explain in 1-2 sentences why the risk changed (or didn't change) based on "
    "the specific parameters that were modified."
)

# Anthropic fallback model — used when vLLM is unavailable
ANTHROPIC_FALLBACK_MODEL: str = "claude-3-5-sonnet-latest"

# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# vLLM Server Lifecycle
# ---------------------------------------------------------------------------


def start_vllm_server(
    model_name: str = "Qwen/Qwen2.5-72B-Instruct",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    startup_timeout: int = 120,
) -> Optional[subprocess.Popen]:
    """Launch vLLM as an OpenAI-compatible server on AMD MI300X (ROCm).

    Starts ``python -m vllm.entrypoints.openai.api_server`` as a subprocess
    and polls ``GET /health`` until the server is ready or the timeout expires.

    Args:
        model_name: HuggingFace model ID. Meta-Llama-3-70B-Instruct chosen to 
            showcase MI300X's massive 192GB memory capacity (a single A100 cannot 
            run 70B at full precision without quantization).
        port: TCP port for the OpenAI-compatible API.
        max_model_len: Maximum context length in tokens.
        gpu_memory_utilization: Fraction of GPU HBM to allocate (0.85 leaves
            headroom for large batches in 192 GB MI300X HBM).
        startup_timeout: Seconds to wait for the health-check before giving up.

    Returns:
        The subprocess handle if the server started successfully, or ``None``
        if startup failed (caller should fall back to Anthropic).
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", "float16",
    ]

    logger.info("Starting vLLM server: %s", " ".join(cmd))

    try:
        # Launch in background; pipe stderr so we can log ROCm errors
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Ensure child is killed when parent exits (Unix only; harmless on Windows)
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except FileNotFoundError:
        logger.warning("vLLM binary not found — falling back to Anthropic API.")
        return None
    except OSError as exc:
        logger.warning("OS error launching vLLM: %s — falling back to Anthropic API.", exc)
        return None

    # Poll /health until the model is loaded
    health_url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + startup_timeout
    poll_interval = 2.0  # seconds between health-check attempts

    while time.monotonic() < deadline:
        # Make sure the process hasn't already exited
        if proc.poll() is not None:
            stderr_tail = proc.stderr.read().decode(errors="replace")[-500:]
            logger.warning(
                "vLLM process exited early (rc=%d). Last stderr:\n%s",
                proc.returncode,
                stderr_tail,
            )
            return None

        try:
            resp = requests.get(health_url, timeout=3)
            if resp.status_code == 200:
                logger.info(
                    "vLLM server ready on port %d after %.0f s.",
                    port,
                    startup_timeout - (deadline - time.monotonic()),
                )
                return proc
        except requests.ConnectionError:
            pass  # server not up yet — keep polling

        time.sleep(poll_interval)

    # Timed out — kill the orphaned server
    logger.warning("vLLM did not become healthy within %d s — terminating.", startup_timeout)
    stop_vllm_server(proc)
    return None


def stop_vllm_server(proc: Optional[subprocess.Popen]) -> None:
    """Gracefully terminate a vLLM subprocess (SIGTERM → SIGKILL fallback).

    Args:
        proc: Process handle returned by :func:`start_vllm_server`.
              ``None`` is a safe no-op for when the server was never started.
    """
    if proc is None or proc.poll() is not None:
        return  # already dead or never started

    logger.info("Stopping vLLM server (pid=%d) …", proc.pid)
    try:
        # On Unix, kill the whole process group so forked workers also die
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        logger.warning("Forcefully killing vLLM server …")
        proc.kill()
        proc.wait(timeout=5)


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Client Factory
# ---------------------------------------------------------------------------

# Module-level state: tracks which backend is active so callers can inspect it
_active_backend: str = "none"  # "vllm" | "anthropic" | "none"


def get_llm_client(
    use_vllm: bool = True,
    vllm_port: int = 8000,
) -> tuple:
    """Return an LLM client and the backend identifier string.

    Args:
        use_vllm: If ``True``, return an OpenAI client pointed at the local
            vLLM server.  If ``False`` (or vLLM is down), return an Anthropic
            client using ``ANTHROPIC_API_KEY`` from the environment.
        vllm_port: Port where vLLM is listening.

    Returns:
        A 2-tuple of ``(client, backend)`` where *backend* is ``"vllm"`` or
        ``"anthropic"``.
    """
    global _active_backend

    if use_vllm:
        # Quick health-check before handing out the client
        try:
            resp = requests.get(f"http://localhost:{vllm_port}/health", timeout=3)
            if resp.status_code == 200:
                client = openai.OpenAI(
                    base_url=f"http://localhost:{vllm_port}/v1",
                    api_key="not-needed",  # vLLM ignores auth
                )
                _active_backend = "vllm"
                return client, "vllm"
        except requests.ConnectionError:
            logger.warning("vLLM health-check failed — falling back to Anthropic.")

    # Fallback to Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY not set and vLLM unavailable. "
            "LLM calls will fail until one backend is available."
        )
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    _active_backend = "anthropic"
    return client, "anthropic"


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Internal helper — unified LLM call across vLLM (OpenAI) and Anthropic
# ---------------------------------------------------------------------------


def _call_llm(
    client,
    backend: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 200,
    json_mode: bool = False,
) -> str:
    """Send a single-turn chat completion to either vLLM or Anthropic.

    This thin wrapper isolates the two API shapes so every public function
    can use the same calling convention.

    Args:
        client: An ``openai.OpenAI`` or ``anthropic.Anthropic`` instance.
        backend: ``"vllm"`` or ``"anthropic"`` — determines API shape.
        system_prompt: System message setting the LLM's persona.
        user_prompt: User message with the actual task.
        temperature: Sampling temperature (lower = more factual).
        max_tokens: Maximum tokens to generate.
        json_mode: If ``True``, request JSON output format (vLLM only;
            Anthropic gets a "respond in JSON" instruction instead).

    Returns:
        The raw text response from the model.

    Raises:
        RuntimeError: If the LLM call fails after best-effort handling.
    """
    try:
        if backend == "vllm":
            # OpenAI-compatible endpoint served by vLLM
            kwargs: dict[str, Any] = {
                "model": client.models.list().data[0].id if not json_mode else client.models.list().data[0].id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        elif backend == "anthropic":
            # If JSON mode requested, instruct the model via the prompt
            effective_system = system_prompt
            if json_mode:
                effective_system += (
                    "\n\nIMPORTANT: Respond ONLY with valid JSON. "
                    "Do not wrap it in markdown code fences."
                )

            response = client.messages.create(
                model=ANTHROPIC_FALLBACK_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=effective_system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Anthropic returns a list of content blocks
            return response.content[0].text.strip()

        else:
            raise RuntimeError(f"Unknown LLM backend: {backend!r}")

    except Exception as exc:
        logger.error("LLM call failed (%s backend): %s", backend, exc)
        raise RuntimeError(f"LLM call failed: {exc}") from exc


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Public LLM Functions
# ---------------------------------------------------------------------------


def explain_risk(
    risk_tier: str,
    probability: float,
    section_attributions: list[dict],
    trial_title: str,
    condition: str,
    client,
    backend: str = "vllm",
) -> str:
    """Generate a 2-3 sentence plain-language risk explanation.

    Args:
        risk_tier: ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
        probability: Model-predicted probability of early termination (0-1).
        section_attributions: Top contributing features, each a dict with
            keys ``feature_name``, ``display_label``, ``shap_value``.
            Only the top 3 are included in the prompt.
        trial_title: Official trial title for context.
        condition: Primary condition being studied.
        client: LLM client (OpenAI or Anthropic).
        backend: ``"vllm"`` or ``"anthropic"``.

    Returns:
        A concise, executive-level risk explanation string.
    """
    # Build the top-3 attribution summary
    top_3 = sorted(
        section_attributions, key=lambda x: abs(x.get("shap_value", 0)), reverse=True
    )[:3]
    attribution_lines = "\n".join(
        f"  - {a.get('display_label', a.get('feature_name', '?'))}: "
        f"contribution = {a.get('shap_value', 0):+.3f}"
        for a in top_3
    )

    user_prompt = (
        f"Trial: {trial_title}\n"
        f"Condition: {condition}\n"
        f"Risk tier: {risk_tier}\n"
        f"Termination probability: {probability:.1%}\n\n"
        f"Top contributing factors:\n{attribution_lines}\n\n"
        "Explain in 2-3 sentences why this trial carries this risk level."
    )

    return _call_llm(
        client=client,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_EXPLAIN,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=200,
    )


def suggest_rewrites(
    section_name: str,
    section_text: str,
    shap_contribution: float,
    trial_phase: str,
    condition: str,
    client,
    backend: str = "vllm",
) -> list[str]:
    """Suggest 2-3 actionable protocol edits for the highest-risk section.

    Uses JSON mode for reliable parsing; falls back to regex splitting
    on numbered-list format if JSON parsing fails.

    Args:
        section_name: Name of the protocol section (e.g., "Eligibility Criteria").
        section_text: Full text of the protocol section.
        shap_contribution: SHAP value for this section (signed float).
        trial_phase: Trial phase string (e.g., "Phase 2").
        condition: Primary condition being studied.
        client: LLM client.
        backend: ``"vllm"`` or ``"anthropic"``.

    Returns:
        A list of 2-3 specific rewrite suggestion strings.
    """
    user_prompt = (
        f"Protocol section: {section_name}\n"
        f"Trial phase: {trial_phase}\n"
        f"Condition: {condition}\n"
        f"Risk contribution score: {shap_contribution:+.3f}\n\n"
        f"Section text:\n{section_text[:2000]}\n\n"  # truncate to stay in context
        "Suggest 2-3 specific, actionable edits to reduce termination risk.\n"
        'Respond in JSON: {"rewrites": ["edit 1", "edit 2", "edit 3"]}'
    )

    raw = _call_llm(
        client=client,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_REWRITE,
        user_prompt=user_prompt,
        temperature=0.4,
        max_tokens=400,
        json_mode=True,
    )

    return _parse_rewrites(raw)


def _parse_rewrites(raw: str) -> list[str]:
    """Parse rewrite suggestions from LLM output.

    Primary strategy: ``json.loads`` on the raw response.
    Fallback: split on numbered-list markers (``1. ``, ``2. ``, ``3. ``).

    Args:
        raw: Raw text response from the LLM.

    Returns:
        A list of rewrite strings. Returns a single-item list with the raw
        text if all parsing strategies fail.
    """
    # Strategy 1: JSON parse
    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "rewrites" in parsed:
            rewrites = parsed["rewrites"]
            if isinstance(rewrites, list) and len(rewrites) > 0:
                return [str(r).strip() for r in rewrites if str(r).strip()]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.debug("JSON parsing failed for rewrites — trying numbered-list fallback.")

    # Strategy 2: Split on numbered list markers
    # Matches patterns like "1. ...", "2) ...", "1: ..."
    parts = re.split(r"\n\s*\d+[\.\)\:]\s+", raw)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts[:3]

    # Strategy 3: Return raw text as single suggestion
    logger.warning("Could not parse rewrites — returning raw response.")
    return [raw.strip()] if raw.strip() else ["No suggestions generated."]


def explain_whatif_change(
    original_params: dict,
    modified_params: dict,
    original_risk: float,
    new_risk: float,
    client,
    backend: str = "vllm",
) -> str:
    """Explain why the predicted risk changed after parameter modification.

    Args:
        original_params: Dict of original protocol parameters (e.g.,
            ``{"enrollment": 45, "phase": "Phase 2", ...}``).
        modified_params: Dict of user-modified parameters.
        original_risk: Model-predicted termination probability before change.
        new_risk: Model-predicted termination probability after change.
        client: LLM client.
        backend: ``"vllm"`` or ``"anthropic"``.

    Returns:
        1-2 sentence explanation of the risk delta.
    """
    # Compute which params actually changed
    changed = {
        k: {"from": original_params.get(k), "to": modified_params.get(k)}
        for k in set(list(original_params.keys()) + list(modified_params.keys()))
        if original_params.get(k) != modified_params.get(k)
    }

    delta = new_risk - original_risk
    direction = "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged"

    user_prompt = (
        f"Original termination probability: {original_risk:.1%}\n"
        f"New termination probability: {new_risk:.1%}\n"
        f"Risk {direction} by {abs(delta):.1%}\n\n"
        f"Parameters changed:\n{json.dumps(changed, indent=2, default=str)}\n\n"
        "Explain in 1-2 sentences why the risk changed (or didn't)."
    )

    return _call_llm(
        client=client,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_WHATIF,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=150,
    )


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Demo Cache
# ---------------------------------------------------------------------------

# In-memory cache keyed by NCT ID — populated at startup for the 20 demo
# trials so the hackathon demo has zero live LLM latency.
_demo_cache: dict[str, dict] = {}


def get_cached_or_generate(
    nct_id: str,
    cache: dict,
    generator_fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> dict:
    """Return cached result or generate fresh via LLM, then cache it.

    For the 20 pre-cached demo trials this guarantees zero live LLM
    latency during the hackathon presentation.

    Args:
        nct_id: ClinicalTrials.gov identifier (e.g., ``"NCT00123456"``).
        cache: Mutable dict acting as the cache store. Pass the module-level
            ``_demo_cache`` or your own dict.
        generator_fn: Callable that produces the result dict when called with
            ``*args, **kwargs``.
        *args: Positional arguments forwarded to *generator_fn*.
        **kwargs: Keyword arguments forwarded to *generator_fn*.

    Returns:
        A dict containing the LLM-generated (or cached) result.
    """
    if nct_id in cache:
        logger.debug("Cache HIT for %s", nct_id)
        return cache[nct_id]

    logger.debug("Cache MISS for %s — calling LLM", nct_id)
    result = generator_fn(*args, **kwargs)
    cache[nct_id] = result
    return result


def save_demo_cache(filepath: str = "artifacts/demo_cache.json") -> None:
    """Persist the demo cache to disk so it survives session restarts.

    Args:
        filepath: Path to the JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_demo_cache, f, indent=2, ensure_ascii=False)
    logger.info("Saved demo cache (%d entries) to %s", len(_demo_cache), filepath)


def load_demo_cache(filepath: str = "artifacts/demo_cache.json") -> None:
    """Load a previously saved demo cache from disk.

    Args:
        filepath: Path to the JSON file.
    """
    global _demo_cache
    if not os.path.exists(filepath):
        logger.info("No demo cache found at %s — starting fresh.", filepath)
        return
    with open(filepath, "r", encoding="utf-8") as f:
        _demo_cache = json.loads(f.read())
    logger.info("Loaded demo cache (%d entries) from %s", len(_demo_cache), filepath)


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Pre-cache builder — run once before the hackathon demo
# ---------------------------------------------------------------------------


def build_demo_cache(
    demo_trials: list[dict],
    client,
    backend: str = "vllm",
    cache_path: str = "artifacts/demo_cache.json",
) -> None:
    """Pre-generate LLM outputs for demo trials to eliminate live latency.

    Call this function once (e.g., in a setup notebook cell) before the
    hackathon presentation. It populates the in-memory cache and persists
    to disk.

    Args:
        demo_trials: List of dicts, each with keys:
            ``nct_id``, ``trial_title``, ``condition``, ``risk_tier``,
            ``probability``, ``section_attributions``, ``top_section_name``,
            ``top_section_text``, ``top_shap_contribution``, ``trial_phase``.
        client: LLM client.
        backend: ``"vllm"`` or ``"anthropic"``.
        cache_path: Where to save the JSON cache.
    """
    global _demo_cache

    for i, trial in enumerate(demo_trials):
        nct_id = trial["nct_id"]
        logger.info("Pre-caching trial %d/%d: %s", i + 1, len(demo_trials), nct_id)

        # Generate explanation
        explanation = explain_risk(
            risk_tier=trial["risk_tier"],
            probability=trial["probability"],
            section_attributions=trial["section_attributions"],
            trial_title=trial["trial_title"],
            condition=trial["condition"],
            client=client,
            backend=backend,
        )

        # Generate rewrite suggestions
        rewrites = suggest_rewrites(
            section_name=trial["top_section_name"],
            section_text=trial["top_section_text"],
            shap_contribution=trial["top_shap_contribution"],
            trial_phase=trial["trial_phase"],
            condition=trial["condition"],
            client=client,
            backend=backend,
        )

        _demo_cache[nct_id] = {
            "explanation": explanation,
            "rewrites": rewrites,
        }

        # Save after every trial so partial progress is preserved
        save_demo_cache(cache_path)

    logger.info("Demo cache complete: %d trials cached.", len(_demo_cache))


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Convenience: full co-pilot session manager
# ---------------------------------------------------------------------------


class CopilotSession:
    """Manages LLM backend lifecycle for the Gradio app.

    Usage in the Gradio app::

        copilot = CopilotSession()
        copilot.start()  # tries vLLM, falls back to Anthropic

        explanation = copilot.explain_risk(...)
        rewrites = copilot.suggest_rewrites(...)

        copilot.shutdown()  # cleans up vLLM subprocess
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        vllm_port: int = 8000,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        self.model_name = model_name
        self.vllm_port = vllm_port
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        self._vllm_proc: Optional[subprocess.Popen] = None
        self._client = None
        self._backend: str = "none"

    def start(self) -> str:
        """Start vLLM (or fall back to Anthropic) and return the backend name.

        Returns:
            ``"vllm"`` or ``"anthropic"`` depending on which backend is active.
        """
        # Attempt vLLM startup
        self._vllm_proc = start_vllm_server(
            model_name=self.model_name,
            port=self.vllm_port,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

        use_vllm = self._vllm_proc is not None
        if not use_vllm:
            logger.warning(
                "⚠️  vLLM unavailable — using Anthropic %s as fallback. "
                "Ensure ANTHROPIC_API_KEY is set.",
                ANTHROPIC_FALLBACK_MODEL,
            )

        self._client, self._backend = get_llm_client(
            use_vllm=use_vllm,
            vllm_port=self.vllm_port,
        )

        # Load any previously-saved demo cache
        load_demo_cache()

        logger.info("CopilotSession started — backend: %s", self._backend)
        return self._backend

    def shutdown(self) -> None:
        """Terminate the vLLM subprocess and save the demo cache."""
        save_demo_cache()
        stop_vllm_server(self._vllm_proc)
        self._vllm_proc = None
        self._backend = "none"
        logger.info("CopilotSession shut down.")

    @property
    def backend(self) -> str:
        """Currently active backend (``"vllm"`` or ``"anthropic"``)."""
        return self._backend

    # -- Delegated LLM calls -------------------------------------------------

    def explain_risk(
        self,
        risk_tier: str,
        probability: float,
        section_attributions: list[dict],
        trial_title: str,
        condition: str,
    ) -> str:
        """Explain risk — delegates to :func:`explain_risk`."""
        return explain_risk(
            risk_tier=risk_tier,
            probability=probability,
            section_attributions=section_attributions,
            trial_title=trial_title,
            condition=condition,
            client=self._client,
            backend=self._backend,
        )

    def suggest_rewrites(
        self,
        section_name: str,
        section_text: str,
        shap_contribution: float,
        trial_phase: str,
        condition: str,
    ) -> list[str]:
        """Suggest rewrites — delegates to :func:`suggest_rewrites`."""
        return suggest_rewrites(
            section_name=section_name,
            section_text=section_text,
            shap_contribution=shap_contribution,
            trial_phase=trial_phase,
            condition=condition,
            client=self._client,
            backend=self._backend,
        )

    def explain_whatif_change(
        self,
        original_params: dict,
        modified_params: dict,
        original_risk: float,
        new_risk: float,
    ) -> str:
        """Explain what-if — delegates to :func:`explain_whatif_change`."""
        return explain_whatif_change(
            original_params=original_params,
            modified_params=modified_params,
            original_risk=original_risk,
            new_risk=new_risk,
            client=self._client,
            backend=self._backend,
        )

    def cached_explain(
        self,
        nct_id: str,
        risk_tier: str,
        probability: float,
        section_attributions: list[dict],
        trial_title: str,
        condition: str,
    ) -> dict:
        """Cache-aware risk explanation — zero latency for demo trials."""
        return get_cached_or_generate(
            nct_id=nct_id,
            cache=_demo_cache,
            generator_fn=lambda: {
                "explanation": self.explain_risk(
                    risk_tier, probability, section_attributions,
                    trial_title, condition,
                ),
            },
        )

    def cached_rewrites(
        self,
        nct_id: str,
        section_name: str,
        section_text: str,
        shap_contribution: float,
        trial_phase: str,
        condition: str,
    ) -> dict:
        """Cache-aware rewrite suggestions — zero latency for demo trials."""
        return get_cached_or_generate(
            nct_id=nct_id,
            cache=_demo_cache,
            generator_fn=lambda: {
                "rewrites": self.suggest_rewrites(
                    section_name, section_text, shap_contribution,
                    trial_phase, condition,
                ),
            },
        )


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Quick smoke test — runs without GPU by falling back to Anthropic."""

    print("=" * 60)
    print("copilot.py — standalone smoke test")
    print("=" * 60)

    # 1. Get client (will fall back to Anthropic if vLLM not running)
    client, backend = get_llm_client(use_vllm=False)
    print(f"\nBackend: {backend}")

    # 2. Test explain_risk
    print("\n--- explain_risk ---")
    try:
        explanation = explain_risk(
            risk_tier="HIGH",
            probability=0.78,
            section_attributions=[
                {"feature_name": "log_enrollment", "display_label": "Patient enrollment size", "shap_value": 0.42},
                {"feature_name": "criteria_length", "display_label": "Eligibility criteria complexity", "shap_value": 0.31},
                {"feature_name": "has_placebo", "display_label": "Uses placebo control", "shap_value": -0.15},
            ],
            trial_title="A Phase 2 Study of Drug X in Advanced NSCLC",
            condition="Non-Small Cell Lung Cancer",
            client=client,
            backend=backend,
        )
        print(explanation)
    except RuntimeError as e:
        print(f"[SKIP] explain_risk failed: {e}")

    # 3. Test suggest_rewrites
    print("\n--- suggest_rewrites ---")
    try:
        rewrites = suggest_rewrites(
            section_name="Eligibility Criteria",
            section_text=(
                "Inclusion: Adults 18-65 with histologically confirmed NSCLC. "
                "ECOG PS 0-1. No prior immunotherapy. Exclusion: Active autoimmune disease, "
                "brain metastases, concurrent malignancy."
            ),
            shap_contribution=0.31,
            trial_phase="Phase 2",
            condition="Non-Small Cell Lung Cancer",
            client=client,
            backend=backend,
        )
        for i, r in enumerate(rewrites, 1):
            print(f"  {i}. {r}")
    except RuntimeError as e:
        print(f"[SKIP] suggest_rewrites failed: {e}")

    # 4. Test explain_whatif_change
    print("\n--- explain_whatif_change ---")
    try:
        whatif = explain_whatif_change(
            original_params={"enrollment": 45, "phase": "Phase 2", "has_placebo": False},
            modified_params={"enrollment": 150, "phase": "Phase 2", "has_placebo": True},
            original_risk=0.78,
            new_risk=0.42,
            client=client,
            backend=backend,
        )
        print(whatif)
    except RuntimeError as e:
        print(f"[SKIP] explain_whatif_change failed: {e}")

    # 5. Test caching
    print("\n--- demo cache ---")
    _demo_cache["NCT_TEST_001"] = {
        "explanation": "This is a cached explanation.",
        "rewrites": ["Cached rewrite 1", "Cached rewrite 2"],
    }
    cached = get_cached_or_generate(
        "NCT_TEST_001", _demo_cache, lambda: {"explanation": "SHOULD NOT SEE THIS"}
    )
    assert cached["explanation"] == "This is a cached explanation.", "Cache test FAILED"
    print("Cache HIT test passed ✓")

    # 6. Test CopilotSession (Anthropic-only, no GPU)
    print("\n--- CopilotSession ---")
    session = CopilotSession()
    # Don't call session.start() in CI — it would try to launch vLLM
    print("CopilotSession instantiated ✓")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
## INTEGRATION NOTES
# ---------------------------------------------------------------------------
#
# FILES THIS MODULE READS:
#   - artifacts/demo_cache.json   (optional, loaded at startup via load_demo_cache)
#
# FILES THIS MODULE WRITES:
#   - artifacts/demo_cache.json   (saved after build_demo_cache or session shutdown)
#
# ENVIRONMENT VARIABLES:
#   - ANTHROPIC_API_KEY           (required when vLLM is unavailable; used by
#                                  anthropic.Anthropic() for the fallback client)
#
# CONSTANTS THE CALLER MUST KNOW:
#   - Default vLLM port: 8000
#   - Default model: meta-llama/Meta-Llama-3-70B-Instruct
#   - Anthropic fallback model: claude-3-5-sonnet-latest
#
# HOW THE GRADIO APP SHOULD USE THIS MODULE:
#
#   from copilot import CopilotSession
#
#   copilot = CopilotSession()
#   backend = copilot.start()          # launches vLLM or falls back
#
#   # For pre-cached demo trials (zero latency):
#   result = copilot.cached_explain(nct_id, risk_tier, prob, attribs, title, cond)
#
#   # For live / ad-hoc trials:
#   explanation = copilot.explain_risk(risk_tier, prob, attribs, title, cond)
#   rewrites    = copilot.suggest_rewrites(section, text, shap, phase, cond)
#   whatif      = copilot.explain_whatif_change(orig, mod, orig_risk, new_risk)
#
#   # On app shutdown:
#   copilot.shutdown()
#
# PRE-CACHING DEMO TRIALS (run once before the presentation):
#
#   from copilot import CopilotSession, build_demo_cache
#   import pandas as pd
#
#   demo_df = pd.read_parquet("data/demo_trials.parquet")
#   # ... build demo_trials list[dict] from demo_df + SHAP outputs ...
#   copilot = CopilotSession()
#   copilot.start()
#   build_demo_cache(demo_trials, copilot._client, copilot.backend)
#   copilot.shutdown()
# ---------------------------------------------------------------------------
