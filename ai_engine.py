import os
import logging
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

PERSONAS = {
    "assistant": {
        "name": "General Assistant",
        "emoji": "🤖",
        "system": "You are Company GPT, a professional internal knowledge assistant. Answer questions clearly, accurately, and professionally based only on provided company documents.",
        "tone": "professional and neutral",
        "description": "Balanced, professional answers for all topics.",
    },
    "hr": {
        "name": "HR Advisor",
        "emoji": "👥",
        "system": "You are an HR Advisor for the company. You specialize in HR policies, benefits, leave management, onboarding, performance reviews, and employee relations. Speak warmly but professionally.",
        "tone": "empathetic and HR-focused",
        "description": "Specialized in HR policies, benefits, and employee matters.",
    },
    "legal": {
        "name": "Legal & Compliance",
        "emoji": "⚖️",
        "system": "You are a Legal & Compliance assistant. You help employees understand company policies, compliance requirements, contracts, and regulatory guidelines. Always recommend consulting a qualified lawyer for legal advice.",
        "tone": "precise, cautious, and compliance-oriented",
        "description": "Policies, compliance, contracts, and regulatory guidelines.",
    },
    "it": {
        "name": "IT Support",
        "emoji": "💻",
        "system": "You are an IT Support assistant. You help employees with technical issues, software tools, system access, security policies, and technology guidelines from the company knowledge base.",
        "tone": "technical yet approachable",
        "description": "Tech tools, system access, IT policies, and security.",
    },
    "finance": {
        "name": "Finance Advisor",
        "emoji": "💰",
        "system": "You are a Finance & Expense assistant. You help employees understand expense policies, reimbursement processes, budgets, payroll, and financial guidelines.",
        "tone": "accurate and detail-oriented",
        "description": "Expenses, reimbursements, payroll, and financial policies.",
    },
    "executive": {
        "name": "Executive Briefing",
        "emoji": "📊",
        "system": "You are an Executive Briefing assistant. You provide concise, high-level summaries and strategic insights from company documents. Format answers as executive briefs with key takeaways.",
        "tone": "concise, strategic, and executive-level",
        "description": "High-level summaries and strategic insights for leadership.",
    },
}


class AIEngine:

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.client  = None
        self._init_model()

    def _init_model(self):
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set.")
            return
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✅ Groq AI ready! Key: {self.api_key[:8]}...")
        except Exception as exc:
            logger.error(f"Groq init failed: {exc}")
            self.client = None

    def get_personas(self):
        return {k: {
            "name": v["name"],
            "emoji": v["emoji"],
            "description": v["description"],
        } for k, v in PERSONAS.items()}

    def generate_answer(self, question: str, context: str, role: str = "viewer", persona: str = "assistant") -> str:
        if self.client is None:
            load_dotenv(override=True)
            self.api_key = os.getenv("GROQ_API_KEY", "").strip()
            if self.api_key:
                self._init_model()

        if self.client is None:
            return "❌ Groq API key not working. Please check your .env file."

        persona_cfg = PERSONAS.get(persona, PERSONAS["assistant"])

        role_note = {
            "admin":  "The user is an Admin with full access.",
            "editor": "The user is an HR Editor.",
            "viewer": "The user is a regular Employee/Viewer.",
        }.get(role, "")

        tone_note = f"Respond in a {persona_cfg['tone']} manner."

        prompt = f"""Answer this employee question using ONLY the company documents below.

{role_note}
{tone_note}

FORMATTING RULES:
- Use bullet points for lists
- Use **bold** for key terms and numbers
- Use clear section headings where needed
- Do NOT include technical artifacts like (cid:127)
- Write clean, professional English
- If the answer is not in the documents, say: "I couldn't find this information in the available documents."

COMPANY DOCUMENTS:
{context}

EMPLOYEE QUESTION:
{question}

YOUR ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": persona_cfg["system"]},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1200,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"Groq error: {exc}")
            return f"❌ Groq API error: {str(exc)}"
