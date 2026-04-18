from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from tokenizers.trainers import BpeTrainer

from data_mixture import DatasetMixture


def log_progress(message: str) -> None:
    print(message, flush=True)


def build_text_iterator(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    max_samples: int | None,
    streaming: bool,
    text_column: str,
):
    dataset = load_dataset(dataset_name, name=dataset_config, split=split, streaming=streaming)
    if streaming:
        seen = 0
        for row in dataset:
            yield row[text_column]
            seen += 1
            if max_samples is not None and seen >= max_samples:
                break
    else:
        if max_samples is not None:
            max_samples = min(max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))
        for row in dataset:
            yield row[text_column]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on text data.")
    parser.add_argument("--mixture-config", default=None)
    parser.add_argument("--dataset-name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="artifacts/tokenizer")
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    source_counts: dict[str, int] = {}

    if args.mixture_config:
        mixture = DatasetMixture(args.mixture_config, total_examples=args.max_samples)
        source_counts = {spec.name: 0 for spec in mixture.sources}

        def training_iterator():
            yielded = 0
            for example in mixture:
                source_counts[example["source"]] += 1
                yielded += 1
                if yielded == 1 or yielded % args.progress_interval == 0:
                    log_progress(
                        f"Tokenizer samples collected: {yielded}/{args.max_samples} | "
                        f"source={example['source']}"
                    )
                yield example["text"]

        iterator = training_iterator()
    else:
        mixture = None

        def training_iterator():
            yielded = 0
            for text in build_text_iterator(
                args.dataset_name,
                args.dataset_config,
                args.split,
                args.max_samples,
                args.streaming,
                args.text_column,
            ):
                yielded += 1
                if yielded == 1 or yielded % args.progress_interval == 0:
                    log_progress(f"Tokenizer samples collected: {yielded}/{args.max_samples}")
                yield text

        iterator = training_iterator()

    tokenizer.train_from_iterator(
        iterator,
        trainer=trainer,
        length=args.max_samples,
    )

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    meta = {
        "source_mode": "mixture" if mixture is not None else "single_dataset",
        "mixture_config": args.mixture_config,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "max_samples": args.max_samples,
        "streaming": args.streaming,
        "text_column": args.text_column,
        "special_tokens": {token: tokenizer.token_to_id(token) for token in special_tokens},
        "tokenizer_path": str(tokenizer_path),
    }
    if mixture is not None:
        meta["mixture"] = mixture.summary()
        meta["source_counts"] = source_counts
    else:
        meta["dataset_name"] = args.dataset_name
        meta["dataset_config"] = args.dataset_config
        meta["split"] = args.split
    (output_dir / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2))

    log_progress(f"Saved tokenizer to {tokenizer_path}")
    log_progress(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
