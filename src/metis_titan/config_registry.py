from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import ChatDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

from .dataloaders import MemmapTokenDataLoader
from .model_registry import model_registry


def _chat_sample_processor(sample):
    return sample["messages"]


def _float8_converters() -> ModelConvertersContainer.Config:
    return ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["output"],
            ),
        ],
    )


def _common_parallelism() -> ParallelismConfig:
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
    )


def metis12_base() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="artifacts/metis12_hf_assets",
        model_spec=model_registry("metis12"),
        model_converters=_float8_converters(),
        optimizer=OptimizersContainer.Config(
            lr=2e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=244,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            global_batch_size=256,
            seq_len=1024,
            steps=15260,
            dtype="bfloat16",
        ),
        dataloader=MemmapTokenDataLoader.Config(
            dataset_path="data/metis12_base",
            split="train",
            infinite=True,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=True,
        ),
        parallelism=_common_parallelism(),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoints/metis12_base",
            interval=1000,
            keep_latest_k=3,
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
        ),
        compile=CompileConfig(enable=False),
        validator=Validator.Config(
            enable=True,
            freq=500,
            steps=100,
            dataloader=MemmapTokenDataLoader.Config(
                dataset_path="data/metis12_base",
                split="val",
                infinite=False,
            ),
        ),
    )


def metis12_chat() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="artifacts/metis12_hf_assets",
        model_spec=model_registry("metis12"),
        model_converters=_float8_converters(),
        optimizer=OptimizersContainer.Config(
            lr=8e-5,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=140,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            global_batch_size=128,
            seq_len=768,
            steps=4688,
            dtype="bfloat16",
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": "data/metis12_chat_sft/train.jsonl",
                "split": "train",
            },
            sample_processor=_chat_sample_processor,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=True,
        ),
        parallelism=_common_parallelism(),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoints/metis12_chat",
            interval=250,
            keep_latest_k=3,
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
        ),
        compile=CompileConfig(enable=False),
        validator=Validator.Config(
            enable=True,
            freq=250,
            steps=50,
            dataloader=ChatDataLoader.Config(
                dataset_path="json",
                load_dataset_kwargs={
                    "data_files": "data/metis12_chat_sft/val.jsonl",
                    "split": "train",
                },
                sample_processor=_chat_sample_processor,
                infinite=False,
            ),
        ),
    )


def metis12_think() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="artifacts/metis12_hf_assets",
        model_spec=model_registry("metis12"),
        model_converters=_float8_converters(),
        optimizer=OptimizersContainer.Config(
            lr=5e-5,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=50,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=12,
            global_batch_size=96,
            seq_len=768,
            steps=1250,
            dtype="bfloat16",
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": "data/metis12_reasoning_sft/train.jsonl",
                "split": "train",
            },
            sample_processor=_chat_sample_processor,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=True,
        ),
        parallelism=_common_parallelism(),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoints/metis12_think",
            interval=100,
            keep_latest_k=3,
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
        ),
        compile=CompileConfig(enable=False),
        validator=Validator.Config(
            enable=True,
            freq=100,
            steps=50,
            dataloader=ChatDataLoader.Config(
                dataset_path="json",
                load_dataset_kwargs={
                    "data_files": "data/metis12_reasoning_sft/val.jsonl",
                    "split": "train",
                },
                sample_processor=_chat_sample_processor,
                infinite=False,
            ),
        ),
    )
