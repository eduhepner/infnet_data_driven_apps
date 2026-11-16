import time

import torch
from fastapi import FastAPI, status
from transformers import (
    GenerationConfig,
    pipeline,
)

from models import Prompt

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pipe_helsinki = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-fr",
    dtype=torch.float32,
    device=device,
    clean_up_tokenization_spaces=False,
)
print("Limite do modelo:", pipe_helsinki.model.config.max_position_embeddings)
print("Max length:", pipe_helsinki.model.generation_config.max_length)

generation_config_helsinki = GenerationConfig(
    do_sample=False,
    max_new_tokens=1024,
)


def questao_1_2(prompt):
    resultado = pipe_helsinki(prompt, generation_config=generation_config_helsinki)
    return resultado[0]["translation_text"]


@app.post("/questao1_2", status_code=status.HTTP_200_OK, response_model=Prompt)
async def questao1_2(prompt: Prompt):
    inicio = time.perf_counter()
    resposta = questao_1_2(prompt.message)
    fim = time.perf_counter()
    print(f"Tempo de execução: {fim - inicio:.4f} segundos")
    return {"message": resposta}
