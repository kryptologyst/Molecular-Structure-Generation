#!/usr/bin/env python3
"""Training script for molecular structure generation."""

import argparse
import logging
from pathlib import Path
import torch
from omegaconf import OmegaConf

from src.models.molecular_models import GraphVAE, GraphGAN, AutoregressiveGenerator
from src.data.dataset import MolecularDataset, load_zinc_dataset, create_data_loaders
from src.training.trainer import MolecularTrainer
from src.utils.utils import get_device, set_seed, setup_logging, create_directories
from src.utils.config import Config

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train molecular generation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--save_dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override with command line arguments
    if args.data_path:
        config.set("data.data_path", args.data_path)
    if args.save_dir:
        config.set("save_dir", args.save_dir)
    if args.device:
        config.set("device", args.device)
    if args.seed:
        config.set("seed", args.seed)
    if args.epochs:
        config.set("training.num_epochs", args.epochs)
    if args.batch_size:
        config.set("training.batch_size", args.batch_size)
    if args.lr:
        config.set("training.learning_rate", args.lr)
    
    # Set up logging
    logger = setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=Path(config.get("log_dir", "logs")) / "training.log"
    )
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Get device
    device = get_device(config.get("device", "auto"))
    logger.info(f"Using device: {device}")
    
    # Create directories
    create_directories(config.get("save_dir", "checkpoints"), [])
    create_directories(config.get("log_dir", "logs"), [])
    
    # Load dataset
    logger.info("Loading dataset...")
    data_path = config.get("data.data_path", "data/")
    dataset_name = config.get("data.dataset", "zinc")
    
    if dataset_name == "zinc":
        smiles_list = load_zinc_dataset(data_path, config.get("data.subset_size"))
    else:
        smiles_list = load_zinc_dataset(data_path, config.get("data.subset_size", 1000))
    
    # Create dataset
    dataset = MolecularDataset(
        smiles_list=smiles_list,
        max_length=config.get("data.max_smiles_length", 100)
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config.get("training.batch_size", 64),
        train_split=config.get("data.train_split", 0.8),
        val_split=config.get("data.val_split", 0.1),
        test_split=config.get("data.test_split", 0.1)
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} molecules")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model_type = config.get("model.type", "graph_vae")
    
    if model_type == "graph_vae":
        model = GraphVAE(
            atom_types=config.get("model.atom_types", 10),
            bond_types=config.get("model.bond_types", 4),
            max_atoms=config.get("model.max_atoms", 50),
            hidden_dim=config.get("model.hidden_dim", 128),
            z_dim=config.get("model.z_dim", 100),
            dropout=config.get("model.dropout", 0.1)
        )
    elif model_type == "graph_gan":
        model = GraphGAN(
            atom_types=config.get("model.atom_types", 10),
            bond_types=config.get("model.bond_types", 4),
            max_atoms=config.get("model.max_atoms", 50),
            hidden_dim=config.get("model.hidden_dim", 128),
            z_dim=config.get("model.z_dim", 100)
        )
    elif model_type == "autoregressive":
        model = AutoregressiveGenerator(
            vocab_size=1000,
            d_model=config.get("model.hidden_dim", 256),
            nhead=8,
            num_layers=6,
            max_length=config.get("data.max_smiles_length", 100),
            dropout=config.get("model.dropout", 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Model created: {model_type}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = MolecularTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config.to_dict()
    )
    
    # Train model
    logger.info("Starting training...")
    num_epochs = config.get("training.num_epochs", 100)
    
    history = trainer.train(
        num_epochs=num_epochs,
        save_dir=config.get("save_dir", "checkpoints")
    )
    
    logger.info("Training completed!")
    logger.info(f"Training time: {history['training_time']:.2f} seconds")
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_metrics = trainer.evaluate_model(
        num_samples=config.get("evaluation.num_samples", 1000)
    )
    
    logger.info("Evaluation results:")
    for metric, value in eval_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save final model
    final_path = Path(config.get("save_dir", "checkpoints")) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'history': history,
        'eval_metrics': eval_metrics
    }, final_path)
    
    logger.info(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()
