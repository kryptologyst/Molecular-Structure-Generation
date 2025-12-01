#!/usr/bin/env python3
"""Sampling script for molecular structure generation."""

import argparse
import logging
from pathlib import Path
import torch
import json

from src.models.molecular_models import GraphVAE, GraphGAN, AutoregressiveGenerator
from src.sampling.sampler import MolecularSampler
from src.evaluation.metrics import MolecularMetrics
from src.utils.utils import get_device, set_seed, setup_logging
from src.utils.config import Config

def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Sample molecules from trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, help="Nucleus sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate generated samples")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(log_level="INFO")
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device or "auto")
    logger.info(f"Using device: {device}")
    
    # Load model checkpoint
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Load configuration
    if args.config:
        config = Config(args.config)
    else:
        config = Config()
        if 'config' in checkpoint:
            config.update(checkpoint['config'])
    
    # Create model
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Model loaded: {model_type}")
    
    # Create sampler
    sampler = MolecularSampler(model, device)
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    
    samples = sampler.sample_molecules(
        num_samples=args.num_samples,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    logger.info(f"Generated {len(samples)} samples")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save samples
    samples_data = []
    for sample in samples:
        samples_data.append({
            'smiles': sample['smiles'],
            'properties': sample['properties']
        })
    
    # Save as JSON
    json_path = output_dir / "generated_samples.json"
    with open(json_path, 'w') as f:
        json.dump(samples_data, f, indent=2)
    
    logger.info(f"Samples saved to {json_path}")
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(samples_data)
    csv_path = output_dir / "generated_samples.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Samples saved to {csv_path}")
    
    # Save as SDF
    try:
        from rdkit.Chem import SDWriter
        sdf_path = output_dir / "generated_samples.sdf"
        writer = SDWriter(str(sdf_path))
        for sample in samples:
            if sample['mol'] is not None:
                writer.write(sample['mol'])
        writer.close()
        logger.info(f"Samples saved to {sdf_path}")
    except Exception as e:
        logger.warning(f"Failed to save SDF: {e}")
    
    # Visualize samples
    try:
        sampler.visualize_samples(samples, str(output_dir / "samples_visualization.png"))
        logger.info(f"Visualization saved to {output_dir / 'samples_visualization.png'}")
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")
    
    # Evaluate samples
    if args.evaluate:
        logger.info("Evaluating generated samples...")
        
        metrics = MolecularMetrics()
        smiles_list = [sample['smiles'] for sample in samples]
        
        eval_metrics = metrics.compute_all_metrics(smiles_list)
        
        # Save evaluation results
        eval_path = output_dir / "evaluation_metrics.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        
        # Print summary
        logger.info("Evaluation Summary:")
        logger.info(f"  Validity: {eval_metrics.get('validity', 0):.4f}")
        logger.info(f"  Uniqueness: {eval_metrics.get('uniqueness', 0):.4f}")
        logger.info(f"  Novelty: {eval_metrics.get('novelty', 0):.4f}")
        logger.info(f"  QED Mean: {eval_metrics.get('qed_mean', 0):.4f}")
        logger.info(f"  LogP Mean: {eval_metrics.get('logp_mean', 0):.4f}")
        logger.info(f"  Lipinski Compliance: {eval_metrics.get('lipinski_compliance', 0):.4f}")
        
        # Create metrics plot
        try:
            metrics.plot_metrics_distribution(smiles_list, str(output_dir / "metrics_distribution.png"))
            logger.info(f"Metrics plot saved to {output_dir / 'metrics_distribution.png'}")
        except Exception as e:
            logger.warning(f"Failed to create metrics plot: {e}")

if __name__ == "__main__":
    main()
