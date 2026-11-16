import time

from fastapi import FastAPI, status
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import PromptTemplate

from models import Prompt

app = FastAPI()

fake_llm = FakeListLLM(
    responses=[
        "Primeira resposta simulada.",
        "Segunda resposta simulada",
        "Terceira resposta simulada",
    ]
)

prompt_fake = PromptTemplate.from_template("Answer the question: {question}.")

chain_fake = prompt_fake | fake_llm


def questao_2_1(mensagem):
    return chain_fake.invoke({"question": mensagem})


@app.post("/questao2_1", status_code=status.HTTP_200_OK, response_model=Prompt)
async def questao2_1(prompt: Prompt):
    inicio = time.perf_counter()
    resposta = questao_2_1(prompt.message)
    fim = time.perf_counter()
    print(f"Tempo de execução: {fim - inicio:.4f} segundos")
    return {"message": resposta}
