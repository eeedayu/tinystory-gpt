import argparse, json, random
from pathlib import Path
import numpy as np, torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
    GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    a=ap.parse_args(); seed_all(a.seed)

    tok=AutoTokenizer.from_pretrained("artifacts/tokenizer")
    ds=load_from_disk("artifacts/tokenized")
    config=GPT2Config(vocab_size=len(tok), n_positions=256, n_ctx=256,
        n_embd=384, n_layer=6, n_head=6, n_inner=1536,
        resid_pdrop=.1, embd_pdrop=.1, attn_pdrop=.1,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, use_cache=False)

    model=GPT2LMHeadModel(config)
    print("params", sum(p.numel() for p in model.parameters()))
    use_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16=torch.cuda.is_available() and not use_bf16

    args=TrainingArguments(output_dir="artifacts/checkpoints", max_steps=a.max_steps,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        gradient_accumulation_steps=8, learning_rate=3e-4, weight_decay=.1,
        warmup_steps=int(a.max_steps * 0.03), lr_scheduler_type="cosine", max_grad_norm=1.0,
        logging_steps=20, eval_strategy="steps", eval_steps=250,
        save_strategy="steps", save_steps=250, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, bf16=use_bf16, fp16=use_fp16,
        gradient_checkpointing=True, report_to=["tensorboard"],
        dataloader_num_workers=4,
        seed=a.seed, data_seed=a.seed)

    collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    trainer=Trainer(model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["validation"], data_collator=collator,
        processing_class=tok)

    trainer.train(resume_from_checkpoint=True if a.resume else None)
    metrics=trainer.evaluate()

    out=Path("artifacts/final"); out.mkdir(parents=True, exist_ok=True)
    trainer.model.config.use_cache=True
    trainer.save_model(out); tok.save_pretrained(out)
    (out/"eval_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()