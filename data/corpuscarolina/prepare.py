import argparse
import hashlib
import os
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
from datasets import load_dataset
from tokenizers import Encoding, Tokenizer
from tqdm import tqdm

Split = Literal["train", "val", "test"]

TAXONOMIES = ("wik", "dat", "leg", "jud", "uni", "soc", "pub")
SPLITS: tuple[Split, ...] = ("train", "val", "test")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "processed"


@dataclass
class SplitStats:
    documents: int = 0
    tokens: int = 0


def ratio(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("a proporção deve estar entre 0 e 1")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("o valor deve ser um inteiro positivo")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baixa e tokeniza o Corpus Carolina em train.bin, val.bin e "
            "test.bin, no mesmo formato usado pelo TinyStories."
        )
    )
    parser.add_argument(
        "--tokenizer-filename",
        default="NousResearch/Llama-2-7b-hf",
        help="Tokenizer no Hugging Face Hub ou caminho local.",
    )
    parser.add_argument(
        "--dataset-filename",
        default="carolina-c4ai/corpus-carolina",
        help="Dataset no Hugging Face Hub ou caminho local.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Revisão do dataset, por exemplo v2.0.1 (padrão: revisão atual).",
    )
    parser.add_argument(
        "--taxonomies",
        nargs="+",
        choices=TAXONOMIES,
        default=list(TAXONOMIES),
        metavar="TAXONOMIA",
        help="Taxonomias que serão processadas (padrão: todas).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Diretório de saída (padrão: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--dtype",
        choices=("int16", "uint16", "int32", "uint32"),
        default="uint16",
        help="Tipo NumPy usado nos arquivos binários.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=8,
        help="Documentos enviados juntos ao tokenizer.",
    )
    parser.add_argument(
        "--val-ratio",
        type=ratio,
        default=0.05,
        help="Proporção de documentos para validação.",
    )
    parser.add_argument(
        "--test-ratio",
        type=ratio,
        default=0.05,
        help="Proporção de documentos para teste.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed do split determinístico.",
    )
    parser.add_argument(
        "--max-documents",
        type=positive_int,
        default=None,
        help="Limite global opcional, útil para testar o pipeline.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Substitui arquivos .bin que já existam.",
    )
    args = parser.parse_args()

    if args.val_ratio + args.test_ratio >= 1:
        parser.error(
            "a soma de --val-ratio e --test-ratio deve ser menor que 1"
        )

    # Evita processar a mesma taxonomia duas vezes se ela for repetida na CLI
    args.taxonomies = list(dict.fromkeys(args.taxonomies))
    return args


def document_split(
    taxonomy: str,
    document_index: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Split:
    """Atribui um documento a um split de modo estável e reprodutível"""
    key = f"{seed}:{taxonomy}:{document_index}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    sample = int.from_bytes(digest, "big") / 2**64

    if sample < test_ratio:
        return "test"
    if sample < test_ratio + val_ratio:
        return "val"
    return "train"


def validate_dtype(tokenizer: Tokenizer, dtype: np.dtype) -> None:
    if dtype.kind not in "iu":
        raise ValueError(f"dtype deve ser inteiro, recebido: {dtype}")

    largest_token_id = max(tokenizer.get_vocab().values())
    if largest_token_id > np.iinfo(dtype).max:
        raise ValueError(
            f"O tokenizer usa o ID {largest_token_id}, que não cabe em {dtype}. "
            "Escolha um dtype maior."
        )


def temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{os.getpid()}.tmp")


def write_encodings(
    batch: list[tuple[Split, str]],
    tokenizer: Tokenizer,
    files: Mapping[Split, BinaryIO],
    dtype: np.dtype,
    stats: dict[Split, SplitStats],
) -> None:
    encodings: list[Encoding] = tokenizer.encode_batch(
        [text for _, text in batch],
        add_special_tokens=True,
    )

    for (split, _), encoding in zip(batch, encodings, strict=True):
        token_ids = np.asarray(encoding.ids, dtype=dtype)
        token_ids.tofile(files[split])
        stats[split].documents += 1
        stats[split].tokens += token_ids.size


def iter_taxonomy(
    dataset_filename: str,
    taxonomy: str,
    revision: str | None,
) -> Iterable[dict]:
    if revision is None:
        return load_dataset(
            dataset_filename,
            taxonomy=taxonomy,
            streaming=True,
            split="corpus",
        )

    return load_dataset(
        dataset_filename,
        taxonomy=taxonomy,
        revision=revision,
        streaming=True,
        split="corpus",
    )


def prepare(args: argparse.Namespace) -> dict[Split, SplitStats]:
    dtype = np.dtype(args.dtype)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = {split: args.out_dir / f"{split}.bin" for split in SPLITS}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Arquivos de saída já existem: {paths}. Use --overwrite para "
            "substituí-los."
        )

    tokenizer = Tokenizer.from_pretrained(args.tokenizer_filename)
    validate_dtype(tokenizer, dtype)

    temporary = {split: temporary_path(path) for split, path in targets.items()}
    stats = {split: SplitStats() for split in SPLITS}
    processed_documents = 0

    try:
        with ExitStack() as stack:
            files = {
                split: stack.enter_context(path.open("wb"))
                for split, path in temporary.items()
            }
            batch: list[tuple[Split, str]] = []

            for taxonomy in args.taxonomies:
                dataset = iter_taxonomy(
                    args.dataset_filename,
                    taxonomy,
                    args.revision,
                )
                progress = tqdm(
                    dataset, desc=f"Tokenizando {taxonomy}", unit="doc"
                )

                for document_index, example in enumerate(progress):
                    text = example["text"]
                    split = document_split(
                        taxonomy,
                        document_index,
                        args.seed,
                        args.val_ratio,
                        args.test_ratio,
                    )
                    batch.append((split, text))
                    processed_documents += 1

                    if len(batch) == args.batch_size:
                        write_encodings(batch, tokenizer, files, dtype, stats)
                        batch.clear()

                    if (
                        args.max_documents is not None
                        and processed_documents >= args.max_documents
                    ):
                        break

                if args.max_documents is not None and (
                    processed_documents >= args.max_documents
                ):
                    break

            if batch:
                write_encodings(batch, tokenizer, files, dtype, stats)

        for split in SPLITS:
            os.replace(temporary[split], targets[split])
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise

    return stats


def print_summary(
    stats: dict[Split, SplitStats],
    out_dir: Path,
    dtype: np.dtype,
) -> None:
    print(f"\nConcluído. Arquivos gravados em {out_dir}:")
    for split in SPLITS:
        split_stats = stats[split]
        size_bytes = split_stats.tokens * dtype.itemsize
        print(
            f"  {split}.bin: {split_stats.documents:,} documentos, "
            f"{split_stats.tokens:,} tokens, {size_bytes / 1024**2:.2f} MiB"
        )


def main() -> None:
    args = parse_args()
    stats = prepare(args)
    print_summary(stats, args.out_dir, np.dtype(args.dtype))


if __name__ == "__main__":
    main()
