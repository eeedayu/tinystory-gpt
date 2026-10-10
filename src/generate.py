import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed



ap=argparse.ArgumentParser(); ap.add_argument("--prompt", default="Once upon a time")
ap.add_argument("--seed", type=int, default=42); a=ap.parse_args(); set_seed(a.seed)

path="artifacts/final"; tok=AutoTokenizer.from_pretrained(path)
dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None

model=AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device).eval()
inputs=tok(a.prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    out=model.generate(**inputs, max_new_tokens=120, do_sample=True,
        temperature=.8, top_p=.95, top_k=50, repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))