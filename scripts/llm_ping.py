"""Quick check that the LLM endpoint works: python scripts/llm_ping.py"""
import logging

from pydantic import BaseModel

from opscopilot.llm.factory import make_llm

logging.basicConfig(level=logging.INFO)


class Capital(BaseModel):
    country: str
    capital: str


llm = make_llm()

print("--- plain chat")
print(llm.chat([{"role": "user", "content": "Spiega in una frase cos'è un database vettoriale."}]))

print("--- structured JSON")
answer = llm.chat_json([{"role": "user", "content": "Qual è la capitale dell'Italia?"}], Capital)
print(answer)
