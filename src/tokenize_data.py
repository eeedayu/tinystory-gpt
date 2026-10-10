import argparse
from datasets import load_dataset
from transformers import AutoTokenizer



ap=argparse.ArgumentParser(); ap.add_argument("--block-size", type=int, default=256)
ap.add_argument("--num-proc", type=int, default=4); a=ap.parse_args()

tok=AutoTokenizer.from_pretrained("artifacts/tokenizer")
assert a.block_size <= tok.model_max_length
raw=load_dataset("json", data_files={"train":"data/train.jsonl",
                                    "validation":"data/valid.jsonl"})

def encode(batch): return tok(batch["text"], add_special_tokens=False, truncation=False)
tokenized=raw.map(encode, batched=True,
    remove_columns=raw["train"].column_names, num_proc=a.num_proc)


def group(batch):
    stream=[]
    for ids in batch["input_ids"]:
        stream.extend(ids); stream.append(tok.eos_token_id)
    usable=(len(stream)//a.block_size)*a.block_size
    chunks=[stream[i:i+a.block_size] for i in range(0, usable, a.block_size)]
    return {"input_ids":chunks,
            "attention_mask":[[1]*a.block_size for _ in chunks]}

if __name__ == "__main__":
    lm=tokenized.map(group, batched=True, batch_size=1000, num_proc=a.num_proc)
    lm.save_to_disk("artifacts/tokenized"); print(lm)