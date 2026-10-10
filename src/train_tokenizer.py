import argparse, json
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]



def texts(path):

    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]

ap = argparse.ArgumentParser()
ap.add_argument("--vocab-size", type=int, default=8000)
args = ap.parse_args()

backend = Tokenizer(models.BPE(unk_token="<unk>"))
backend.normalizer = normalizers.NFKC()
backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
backend.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2,
    special_tokens=SPECIAL,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

backend.train_from_iterator(texts("data/train.jsonl"), trainer=trainer)


out = Path("artifacts/tokenizer"); out.mkdir(parents=True, exist_ok=True)
fast = PreTrainedTokenizerFast(tokenizer_object=backend,
    pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
    unk_token="<unk>", model_max_length=256)
fast.save_pretrained(out)

for text in ["Once upon a time, there was a small robot.",
            "The dragon couldn't open the blue door!"]:
    ids = fast.encode(text, add_special_tokens=False)
    print(len(ids), ids[:20], fast.decode(ids))
print("vocab=", len(fast), "special=", fast.special_tokens_map)