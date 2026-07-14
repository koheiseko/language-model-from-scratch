import argparse
import os

import numpy as np
import pyarrow.compute as pc
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configuração para a tokenização do dataset TinyStories"
    )

    parser.add_argument(
        "--tokenizer_filename", default="NousResearch/Llama-2-7b-hf", type=str
    )
    parser.add_argument(
        "--dataset_filename", default="roneneldan/TinyStories", type=str
    )
    parser.add_argument("--out_dir", default="processed", type=str)
    parser.add_argument(
        "--dtype",
        default="uint16",
        type=str,
        help="Tipo de dado do numpy (int16, uint16, int32, uint32)",
    )
    parser.add_argument("--num_proc", default=8, type=int)
    parser.add_argument("--total_batch", default=1024, type=int)

    args = parser.parse_args()

    return args


def prepare():
    args = parse_args()

    tokenizer = Tokenizer.from_pretrained(args.tokenizer_filename)

    dataset = load_dataset(args.dataset_filename)

    split_dataset = dataset["train"].train_test_split(
        test_size=0.05, seed=42, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")
    split_dataset["test"] = dataset["validation"]

    def process(example):
        ids = tokenizer.encode(example["text"]).ids

        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        desc="Tokenizando os dados",
        num_proc=args.num_proc,
        remove_columns=["text"],
    )

    tokenized.set_format(type="numpy", columns=["ids"], dtype=args.dtype)

    os.makedirs(args.out_dir, exist_ok=True)
    for split, dt in tokenized.items():
        len_arrow = dt.data.column("len")
        arr_len = pc.sum(len_arrow).as_py()

        filename = os.path.join(args.out_dir, f"{split}.bin")

        arr = np.memmap(
            filename=filename, dtype=args.dtype, mode="w+", shape=(arr_len,)
        )

        total_batch = args.total_batch
        idx = 0

        for batch_idx in tqdm(
            range(total_batch), desc=f"Escrevendo em {filename}"
        ):
            batch = dt.shard(
                num_shards=total_batch, index=batch_idx, contiguous=True
            )

            ids = np.concatenate(batch["ids"], dtype=args.dtype)

            arr[idx : idx + len(ids)] = ids

            idx += len(ids)

        arr.flush()


if __name__ == "__main__":
    prepare()
