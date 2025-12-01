#!/usr/bin/env python3
"""Command-line interface for molecular structure generation."""

import argparse
import sys
from pathlib import Path

from src import GraphVAE, GraphGAN, AutoregressiveGenerator
from src.data.dataset import MolecularDataset, load_zinc_dataset
from src.sampling.sampler import MolecularSampler
from src.evaluation.metrics import MolecularMetrics
from src.utils.utils import get_device, set_seed, setup_logging


def train_command(args):
    """Train a molecular generation model."""
    from scripts.train import main as train_main
    
    # Convert args to sys.argv format
    sys.argv = ['train.py'] + [
        f'--{k.replace("_", "-")}' if v is not None else f'--{k.replace("_", "-")}={v}'
        for k, v in vars(args).items() if v is not None
    ]
    
    train_main()


def sample_command(args):
    """Sample molecules from a trained model."""
    from scripts.sample import main as sample_main
    
    # Convert args to sys.argv format
    sys.argv = ['sample.py'] + [
        f'--{k.replace("_", "-")}' if v is not None else f'--{k.replace("_", "-")}={v}'
        for k, v in vars(args).items() if v is not None
    ]
    
    sample_main()


def demo_command(args):
    """Launch the interactive demo."""
    import subprocess
    import sys
    
    demo_path = Path(__file__).parent / "demo" / "streamlit_app.py"
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(demo_path)]
    
    if args.port:
        cmd.extend(["--server.port", str(args.port)])
    
    if args.host:
        cmd.extend(["--server.address", args.host])
    
    subprocess.run(cmd)


def evaluate_command(args):
    """Evaluate molecular samples."""
    import json
    from pathlib import Path
    
    # Load SMILES from file
    if args.input.endswith('.json'):
        with open(args.input, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                if 'smiles' in data[0]:
                    smiles_list = [item['smiles'] for item in data]
                else:
                    smiles_list = data
            else:
                smiles_list = data
    else:
        with open(args.input, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    
    # Compute metrics
    metrics = MolecularMetrics()
    results = metrics.compute_all_metrics(smiles_list)
    
    # Save results
    output_path = Path(args.output) if args.output else Path(args.input).with_suffix('.metrics.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"  Total molecules: {results.get('total_count', 0)}")
    print(f"  Validity: {results.get('validity', 0):.4f}")
    print(f"  Uniqueness: {results.get('uniqueness', 0):.4f}")
    print(f"  Novelty: {results.get('novelty', 0):.4f}")
    print(f"  QED Mean: {results.get('qed_mean', 0):.4f}")
    print(f"  LogP Mean: {results.get('logp_mean', 0):.4f}")
    print(f"  Lipinski Compliance: {results.get('lipinski_compliance', 0):.4f}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Molecular Structure Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  molecular-gen train --epochs 100 --batch-size 64

  # Sample molecules
  molecular-gen sample --model-path checkpoints/best_model.pt --num-samples 100

  # Launch interactive demo
  molecular-gen demo --port 8501

  # Evaluate generated molecules
  molecular-gen evaluate --input samples.json --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a molecular generation model')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--data-path', type=str, help='Path to dataset')
    train_parser.add_argument('--save-dir', type=str, help='Directory to save checkpoints')
    train_parser.add_argument('--device', type=str, help='Device to use')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Sample molecules from trained model')
    sample_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    sample_parser.add_argument('--config', type=str, help='Path to config file')
    sample_parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to generate')
    sample_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    sample_parser.add_argument('--top-k', type=int, help='Top-k sampling')
    sample_parser.add_argument('--top-p', type=float, help='Nucleus sampling')
    sample_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    sample_parser.add_argument('--output-dir', type=str, default='samples', help='Output directory')
    sample_parser.add_argument('--device', type=str, help='Device to use')
    sample_parser.add_argument('--evaluate', action='store_true', help='Evaluate generated samples')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Launch interactive demo')
    demo_parser.add_argument('--port', type=int, default=8501, help='Port for Streamlit app')
    demo_parser.add_argument('--host', type=str, default='localhost', help='Host for Streamlit app')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate molecular samples')
    eval_parser.add_argument('--input', type=str, required=True, help='Input file with SMILES')
    eval_parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging
    logger = setup_logging(log_level="INFO")
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'sample':
        sample_command(args)
    elif args.command == 'demo':
        demo_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
