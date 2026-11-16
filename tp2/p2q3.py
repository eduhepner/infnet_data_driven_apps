import time

from fastapi import FastAPI, status
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from models import Prompt, Translation

app = FastAPI()

pipe = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-de")

llm = HuggingFacePipeline(pipeline=pipe)


# prompt_german = PromptTemplate.from_template("Translate to german: {text}.")
# chain_german = prompt_german | llm


def questao_2_3(mensagem):
    resposta = llm.invoke(mensagem)
    return resposta


@app.post("/questao2_3", status_code=status.HTTP_200_OK, response_model=Translation)
async def questao2_3(prompt: Prompt):
    inicio = time.perf_counter()
    resposta = questao_2_3(prompt.message)
    fim = time.perf_counter()
    print(f"Tempo de execução: {fim - inicio:.4f} segundos")
    return {"translation": resposta}
