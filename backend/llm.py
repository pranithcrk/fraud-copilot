"""LLM explainer using local Ollama (Llama 3)."""

import httpx


class FraudExplainer:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def explain(self, amount: float, score: float, top_features: list[dict]) -> str:
        risk = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.3 else "LOW"
        features_text = ", ".join(
            f"{f['name']}={f['value']} (importance: {f['importance']})"
            for f in top_features
        )

        prompt = f"""You are a fraud analyst AI. A transaction of ${amount:.2f} was scored {score:.2%} fraud probability ({risk} risk).

Top contributing features: {features_text}

Provide a concise 2-3 sentence explanation of why this transaction was flagged at this risk level. Reference the specific features and their values. Be direct and actionable."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                return resp.json()["response"].strip()
        except Exception as e:
            return f"LLM unavailable ({e}). Score: {score:.2%} risk. Key factors: {features_text}"
