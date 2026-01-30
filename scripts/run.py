#!/usr/bin/env python
"""Main entry point for training and evaluation."""

import argparse
from pathlib import Path

import torch

from src.utils.config import load_config, merge_configs
from src.utils.reproducibility import set_seed, get_device
from src.utils.logging import setup_logger
from src.data.factory import data_factory
from src.models.registry import get_model

# Import models to register them
from src.models.transformers import informer, autoformer
from src.models.neural_ode import latent_ode, ode_rnn

from src.training.trainer import StandardTrainer
from src.training.ode_trainer import AdjointTrainer
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Time Series Forecasting Benchmark")

    # Config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to model config file (optional)",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default=None,
        help="Path to data config file (optional)",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "both"],
        help="Run mode",
    )

    # Overrides
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=None)

    # Output
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load and merge configs
    config = load_config(args.config)

    if args.model_config:
        model_config = load_config(args.model_config)
        config = merge_configs(config, model_config)

    if args.data_config:
        data_config = load_config(args.data_config)
        config = merge_configs(config, data_config)

    # Apply command line overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.gpu is not None:
        config["gpu"] = args.gpu

    config.setdefault("logging", {})["save_dir"] = args.save_dir

    # Setup
    set_seed(config.get("seed", 42))
    device = get_device(config.get("gpu"), config.get("use_gpu", True))
    logger = setup_logger(
        log_dir=Path(args.save_dir) / "logs" if args.mode in ["train", "both"] else None
    )

    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    # Create data loaders
    logger.info("Loading data...")
    train_dataset, train_loader = data_factory(config, "train")
    val_dataset, val_loader = data_factory(config, "val")
    test_dataset, test_loader = data_factory(config, "test")

    # Update config with data dimensions
    config["enc_in"] = train_dataset.get_feature_dim()
    config["dec_in"] = train_dataset.get_feature_dim()
    config["c_out"] = train_dataset.get_feature_dim()

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Feature dim: {config['enc_in']}")

    # Create model
    model_name = config.get("model", {}).get("name", "informer")
    logger.info(f"Creating model: {model_name}")
    model = get_model(model_name, config)
    model.to(device)

    param_count = model.get_parameter_count()
    logger.info(f"Model parameters: {param_count:,}")

    # Select trainer based on model type
    if model.model_type == "ode":
        trainer_cls = AdjointTrainer
        logger.info("Using AdjointTrainer for ODE model")
    else:
        trainer_cls = StandardTrainer
        logger.info("Using StandardTrainer")

    # Training
    if args.mode in ["train", "both"]:
        logger.info("Starting training...")
        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )
        history = trainer.fit()

        # Save training history
        import json
        history_path = Path(args.save_dir) / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)
        logger.info(f"Training history saved to: {history_path}")

    # Evaluation
    if args.mode in ["test", "both"]:
        logger.info("Starting evaluation...")
        evaluator = Evaluator(
            model=model,
            device=device,
            pred_len=config.get("pred_len", 96),
        )

        # Evaluate on test set
        results = evaluator.evaluate(test_loader, metrics=["mse", "mae", "rmse"])
        evaluator.print_results(results)

        # Save results
        results_path = Path(args.save_dir) / "test_results.json"
        evaluator.save_results(results, results_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
