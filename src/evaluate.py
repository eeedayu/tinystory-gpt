import json, math, torch
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments)

path="artifacts/final"; tok=AutoTokenizer.from_pretrained(path)
model=AutoModelForCausalLM.from_pretrained(path)
ds=load_from_disk("artifacts/tokenized")["validation"]
args=TrainingArguments(output_dir="artifacts/eval", per_device_eval_batch_size=8,
    report_to="none", bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())

trainer=Trainer(model=model, args=args, eval_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False))

if __name__ == "__main__":
    m=trainer.evaluate()
    m["perplexity"]=math.exp(m["eval_loss"]) if m["eval_loss"] < 20 else float("inf")
    print(json.dumps(m, indent=2))