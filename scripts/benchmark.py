#!/usr/bin/env python
"""Run benchmark experiments across models and datasets."""

import argparse
import json
from pathlib import Path
from datetime import datetime
from itertools import product

import torch

from src.utils.config import load_config, merge_configs
from src.utils.reproducibility import set_seed, get_device
from src.utils.logging import setup_logger
from src.data.factory import data_factory
from src.models.registry import get_model, list_models
from src.training.trainer import StandardTrainer
from src.training.ode_trainer import AdjointTrainer
from src.evaluation.evaluator import Evaluator


def run_experiment(
    model_name: str,
    dataset_name: str,
    base_config_path: str,
    output_dir: Path,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Run a single experiment.

    Args:
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        base_config_path: Path to base config.
        output_dir: Output directory.
        device: Device to run on.
        seed: Random seed.

    Returns:
        Dictionary of results.
    """
    # Load configs
    base_config = load_config(base_config_path)
    model_config = load_config(f"configs/models/{model_name}.yaml")
    data_config = load_config(f"configs/datasets/{dataset_name}.yaml")

    config = merge_configs(base_config, model_config)
    config = merge_configs(config, data_config)
    config["seed"] = seed

    # Setup
    set_seed(seed)
    exp_dir = output_dir / f"{model_name}_{dataset_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(name=f"{model_name}_{dataset_name}", log_dir=exp_dir)
    logger.info(f"Running: {model_name} on {dataset_name}")

    try:
        # Load data
        train_dataset, train_loader = data_factory(config, "train")
        val_dataset, val_loader = data_factory(config, "val")
        test_dataset, test_loader = data_factory(config, "test")

        config["enc_in"] = train_dataset.get_feature_dim()
        config["dec_in"] = train_dataset.get_feature_dim()
        config["c_out"] = train_dataset.get_feature_dim()

        # Create model
        model = get_model(model_name, config)
        model.to(device)

        # Select trainer
        trainer_cls = AdjointTrainer if model.model_type == "ode" else StandardTrainer
        config.setdefault("logging", {})["save_dir"] = str(exp_dir / "checkpoints")

        # Train
        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )
        history = trainer.fit()

        # Evaluate
        evaluator = Evaluator(model, device, config.get("pred_len", 96))
        results = evaluator.evaluate(test_loader, metrics=["mse", "mae", "rmse"])

        # Save results
        results["model"] = model_name
        results["dataset"] = dataset_name
        results["parameters"] = model.get_parameter_count()

        results_path = exp_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results: MSE={results['mse']:.6f}, MAE={results['mae']:.6f}")

        return results

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return {
            "model": model_name,
            "dataset": dataset_name,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run benchmark experiments")

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to benchmark (default: all registered)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["weather", "ili", "exchange"],
        help="Datasets to benchmark",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/base/default.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/benchmark",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.gpu)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Get models
    if args.models is None:
        models = list_models()
    else:
        models = args.models

    print(f"Models: {models}")
    print(f"Datasets: {args.datasets}")

    # Run experiments
    all_results = []
    for model_name, dataset_name in product(models, args.datasets):
        print(f"\n{'='*60}")
        print(f"Experiment: {model_name} on {dataset_name}")
        print(f"{'='*60}")

        results = run_experiment(
            model_name=model_name,
            dataset_name=dataset_name,
            base_config_path=args.base_config,
            output_dir=output_dir,
            device=device,
            seed=args.seed,
        )
        all_results.append(results)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Dataset':<12} {'MSE':>12} {'MAE':>12} {'Params':>12}")
    print("-" * 80)

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<15} {r['dataset']:<12} {'ERROR':>12}")
        else:
            print(
                f"{r['model']:<15} {r['dataset']:<12} "
                f"{r['mse']:>12.6f} {r['mae']:>12.6f} {r['parameters']:>12,}"
            )

    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
