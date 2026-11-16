import os
import time

from fastapi import FastAPI, status
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from models import Prompt, Translation

app = FastAPI()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

prompt_french = PromptTemplate.from_template("Translate to french: {text}.")

chain_french = prompt_french | llm


def questao_2_2(mensagem):
    resposta = chain_french.invoke({"text": mensagem})
    return resposta.content


@app.post("/questao2_2", status_code=status.HTTP_200_OK, response_model=Translation)
async def questao2_2(prompt: Prompt):
    inicio = time.perf_counter()
    resposta = questao_2_2(prompt.message)
    fim = time.perf_counter()
    print(f"Tempo de execução: {fim - inicio:.4f} segundos")
    return {"translation": resposta}
