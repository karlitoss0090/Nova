import json
from typing import List, Dict, AsyncGenerator, Optional
import httpx

from core.config import settings


# ── System prompt ─────────────────────────────────────────────────────────────

NOVA_SYSTEM_PROMPT = """\
Tu es NOVA — l'assistant IA personnel de Karl. Son propre Jarvis.

PROFIL DE KARL:
- Administrateur IT chez Tricentis (compagnie de recyclage au Québec)
- Étudiant en génie logiciel à l'ÉTS Montréal
- Stack tech: Python, PostgreSQL, Next.js, Android/Kotlin, Docker, Microsoft 365
- Projets perso: BudgetApp, homelab, NOVA (toi!)
- Lieu: Québec, Canada

TA PERSONNALITÉ:
- Intelligent, précis, helpful — comme Jarvis mais plus chaleureux
- Quand Karl écrit en français: réponds en québécois informel (utilise "tu", sois naturel)
- Quand Karl écrit en anglais: réponds en anglais
- Karl est technique — pas besoin de sur-expliquer les bases
- Sois concis mais complet. Montre de l'initiative quand c'est pertinent
- Tu n'es pas un assistant générique — tu es le partenaire IA de Karl

TES CAPACITÉS (version actuelle):
- Conversation, raisonnement, aide technique
- Aide aux cours à l'ÉTS (génie logiciel)
- Support IT (Windows, M365, PowerShell, réseau)
- Mémoire des conversations passées et des faits sur Karl
- Interaction vocale (speech-to-text + text-to-speech)

CAPACITÉS À VENIR (bientôt):
- Agents spécialisés: étude, projets, travail, recherche
- Mémoire sémantique vectorielle (ChromaDB)
- Tâches complexes via l'API Claude

RÈGLES:
- Jamais prétendre avoir des capacités que tu n'as pas encore
- Si tu sais pas, dis-le — Karl préfère l'honnêteté à l'hallucination
- Garde un ton conversationnel sauf si Karl demande des explications détaillées
- Tu es NOVA, pas "un assistant IA" générique\
"""


# ── Language detection ────────────────────────────────────────────────────────

_FR_INDICATORS = {
    "je ", "tu ", "il ", "elle ", "nous ", "vous ", "ils ", "elles ",
    "le ", "la ", "les ", "un ", "une ", "des ", "du ", "de la ",
    "c'est", "j'ai", "j'suis", "c'tu", "bonjour", "salut", "merci",
    "oui", "non", "quoi", "comment", "pourquoi", "quand", "où", "ça ",
    "pas ", "avec ", "pour ", "dans ", "sur ", "est-ce", "qu'est",
    "qu'il", "qu'elle", "c'pas", "genre", "fait que", "en tout cas",
    "t'as", "t'es", "m'as", "chu ", "j'va", "esti", "câline",
}


def detect_language(text: str) -> str:
    """Detect whether text is French or English."""
    lower = text.lower()
    score = sum(1 for ind in _FR_INDICATORS if ind in lower)
    return "fr" if score >= 2 else "en"


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self):
        self._http = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        await self._http.aclose()

    def _build_payload(
        self,
        message: str,
        history: List[Dict],
        facts: List[Dict],
        stream: bool,
        agent_system_prompt: Optional[str] = None,
    ) -> dict:
        # Build system prompt
        system = agent_system_prompt or NOVA_SYSTEM_PROMPT
        if facts:
            facts_block = "\n\nMÉMOIRE — FAITS SUR KARL:\n"
            for f in facts:
                facts_block += f"  [{f['category']}] {f['key']}: {f['value']}\n"
            system += facts_block

        # Build messages list from history (last 12 turns for context)
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history[-12:]
        ]
        messages.append({"role": "user", "content": message})

        return {
            "model": settings.OLLAMA_MODEL,
            "system": system,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": settings.OLLAMA_TEMPERATURE,
                "top_p": settings.OLLAMA_TOP_P,
                "num_predict": settings.OLLAMA_NUM_PREDICT,
            },
        }

    async def chat(
        self,
        message: str,
        history: List[Dict],
        facts: List[Dict],
        agent_system_prompt: Optional[str] = None,
    ) -> str:
        payload = self._build_payload(message, history, facts, stream=False, agent_system_prompt=agent_system_prompt)
        resp = await self._http.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def chat_stream(
        self,
        message: str,
        history: List[Dict],
        facts: List[Dict],
        agent_system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        payload = self._build_payload(message, history, facts, stream=True, agent_system_prompt=agent_system_prompt)
        async with self._http.stream(
            "POST",
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if not data.get("done", False):
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

    async def check_availability(self) -> bool:
        try:
            resp = await self._http.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


llm_client = LLMClient()
