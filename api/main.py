"""
api/main.py
FastAPI backend serving both the base model and the DPO fine-tuned model
side by side for comparison.
"""

import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DPO_MODEL_DIR = "dpo-coding-model"

app = FastAPI(title="DPO Preference Alignment API")

state = {"base_model": None, "dpo_model": None, "tokenizer": None}


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    base_response: str
    dpo_response: str
    base_generation_time: float
    dpo_generation_time: float


@app.on_event("startup")
def load_models():
    print("Loading models... this happens once at startup.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    dpo_model = PeftModel.from_pretrained(base_model, DPO_MODEL_DIR)

    state["base_model"] = base_model
    state["dpo_model"] = dpo_model
    state["tokenizer"] = tokenizer
    print("Models loaded.")


def generate(model, tokenizer, prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@app.get("/health")
def health():
    ready = state["base_model"] is not None and state["dpo_model"] is not None
    return {"status": "ok" if ready else "loading", "models_loaded": ready}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    tokenizer = state["tokenizer"]

    start = time.time()
    base_response = generate(state["base_model"], tokenizer, request.prompt)
    base_time = round(time.time() - start, 2)

    start = time.time()
    dpo_response = generate(state["dpo_model"], tokenizer, request.prompt)
    dpo_time = round(time.time() - start, 2)

    return GenerateResponse(
        base_response=base_response,
        dpo_response=dpo_response,
        base_generation_time=base_time,
        dpo_generation_time=dpo_time,
    )
