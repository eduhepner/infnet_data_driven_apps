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


pipe_gpt2 = pipeline(
    "text-generation",
    model="openai-community/gpt2",
    dtype=torch.float32,
    device=device,
    clean_up_tokenization_spaces=False,
)
pipe_gpt2.model.generation_config.max_length = None

generation_config_gpt2 = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_k=100,
    top_p=0.9,
)


def questao_1_1(prompt):
    resultado = pipe_gpt2(prompt, generation_config=generation_config_gpt2)
    return resultado[0]["generated_text"]


@app.post("/questao1_1", status_code=status.HTTP_200_OK, response_model=Prompt)
async def questao1_1(prompt: Prompt):
    inicio = time.perf_counter()
    resposta = questao_1_1(prompt.message)
    fim = time.perf_counter()
    print(f"Tempo de execução: {fim - inicio:.4f} segundos")
    return {"message": resposta}
