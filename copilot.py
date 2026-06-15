import openai, json, logging
log = logging.getLogger(__name__)

CLIENT = None

def get_client(port=8000):
    global CLIENT
    if CLIENT is None:
        CLIENT = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-needed"
        )
    return CLIENT

def get_model_name(port=8000):
    client = get_client(port)
    models = client.models.list()
    return models.data[0].id   # whatever loaded — Qwen or Llama

def explain_risk(
    trial_title: str,
    risk_tier: str,
    probability: float,
    section_attributions: list[dict],
    phase: int,
    enrollment: int,
) -> str:
    top_sections = sorted(
        section_attributions,
        key=lambda x: abs(x["contribution"]),
        reverse=True
    )[:2]
    
    sections_text = "\n".join([
        f"- {s['section']}: {s['contribution']:+.3f} ({s['direction']})"
        for s in top_sections
    ])

    prompt = f"""You are a clinical research expert reviewing a trial risk assessment.

Trial: {trial_title}
Risk prediction: {risk_tier} ({int(probability*100)}% probability of early termination)
Phase: {phase}
Enrollment target: {enrollment} patients

Top risk factors from model:
{sections_text}

In exactly 2-3 sentences, explain to a clinical operations VP why this trial
is at risk. Be specific — name the actual protocol issues. Reference real
failure patterns from clinical trial literature. Do not mention ML or SHAP."""

    client = get_client()
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def suggest_rewrites(
    trial_title: str,
    section_name: str,
    section_text: str,
    risk_contribution: float,
    phase: int,
    condition: str,
) -> list[str]:
    prompt = f"""You are a senior clinical trial protocol editor.

Trial: {trial_title}
Condition: {condition}
Phase: {phase}
High-risk section: {section_name} (risk contribution: {risk_contribution:+.3f})

Section text:
{section_text[:800]}

Give exactly 3 specific, actionable edits to reduce termination risk.
Format as JSON array: ["edit 1", "edit 2", "edit 3"]
Each edit must be specific enough to implement directly.
Reference FDA/ICH guidance where relevant.
Return ONLY the JSON array, no other text."""

    client = get_client()
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )
    
    text = response.choices[0].message.content.strip()
    try:
        # strip markdown fences if present
        text = text.replace("```json","").replace("```","").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: split on newlines
        lines = [l.strip().lstrip("123.-) ") for l in text.split("\n") if l.strip()]
        return lines[:3]


def explain_whatif(
    original_enrollment: int,
    new_enrollment: int,
    original_risk: float,
    new_risk: float,
    phase: int,
) -> str:
    direction = "decreased" if new_risk < original_risk else "increased"
    delta = abs(new_risk - original_risk) * 100

    prompt = f"""Clinical trial what-if analysis:
- Enrollment changed: {original_enrollment} → {new_enrollment} patients
- Risk score {direction} by {delta:.1f} percentage points
- Phase {phase} trial

In 1-2 sentences, explain why this enrollment change affected termination risk.
Be specific about statistical power and Phase {phase} norms."""

    client = get_client()
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def check_server(port=8000) -> bool:
    try:
        get_client(port).models.list()
        return True
    except Exception:
        return False
