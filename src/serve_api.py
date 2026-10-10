from contextlib import asynccontextmanager
import asyncio, os, time, torch, uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer



MODEL=os.getenv("MODEL_PATH", "artifacts/final")
tok=model=device=None; lock=asyncio.Lock()

class GenIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=120, ge=1, le=256)
    temperature: float = Field(default=.8, ge=.05, le=2.0)
    top_p: float = Field(default=.95, gt=0, le=1.0)

class CompletionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=4000)
    max_tokens: int = Field(default=120, ge=1, le=256)
    temperature: float = Field(default=.8, ge=.05, le=2.0)
    top_p: float = Field(default=.95, gt=0, le=1.0)
    stream: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tok, model, device
    tok=AutoTokenizer.from_pretrained(MODEL)
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device=="cuda" else None
    model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.to(device).eval(); yield

app=FastAPI(title="TinyStory GPT", lifespan=lifespan)


@app.get("/health")
def health(): return {"ok": model is not None, "model": MODEL, "device": device}

async def generate_ids(prompt, max_new_tokens, temperature, top_p):
    if model is None: raise HTTPException(503, "model not ready")
    batch=tok(prompt, return_tensors="pt", truncation=True,
            max_length=model.config.n_positions).to(device)
    
    async with lock:
        with torch.inference_mode():
            ids=model.generate(**batch, do_sample=True,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id)
    return batch, ids

@app.post("/generate")
async def generate(x: GenIn):
    _, ids=await generate_ids(x.prompt, x.max_new_tokens, x.temperature, x.top_p)
    return {"text":tok.decode(ids[0], skip_special_tokens=True), "model":MODEL}

@app.post("/v1/completions")
async def completions(x: CompletionIn):
    if x.stream:
        raise HTTPException(400, "streaming is not supported")

    batch, ids=await generate_ids(x.prompt, x.max_tokens, x.temperature, x.top_p)
    prompt_tokens=batch["input_ids"].shape[-1]
    generated_ids=ids[0, prompt_tokens:]
    completion_tokens=generated_ids.numel()
    finish_reason="stop" if tok.eos_token_id in generated_ids.tolist() else "length"

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": x.model,
        "choices": [{
            "text": tok.decode(generated_ids, skip_special_tokens=True),
            "index": 0,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }