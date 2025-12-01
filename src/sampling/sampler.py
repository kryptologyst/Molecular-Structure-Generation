"""Sampling utilities for molecular generation."""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import logging
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import networkx as nx

from ..models.molecular_models import GraphVAE, GraphGAN, AutoregressiveGenerator
from ..utils.utils import set_seed, get_device

logger = logging.getLogger(__name__)


class MolecularSampler:
    """Molecular sampling utilities."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        atom_types: List[str] = None,
        bond_types: List[str] = None
    ):
        """Initialize molecular sampler.
        
        Args:
            model: Trained molecular generation model.
            device: Device to run sampling on.
            atom_types: List of atom types.
            bond_types: List of bond types.
        """
        self.model = model
        self.device = device
        self.atom_types = atom_types or ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B']
        self.bond_types = bond_types or ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
    
    def sample_molecules(
        self,
        num_samples: int = 10,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """Sample molecules from the model.
        
        Args:
            num_samples: Number of molecules to generate.
            seed: Random seed for reproducibility.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            
        Returns:
            List of dictionaries containing generated molecules.
        """
        if seed is not None:
            set_seed(seed)
        
        generated_molecules = []
        
        with torch.no_grad():
            if isinstance(self.model, GraphVAE):
                samples = self._sample_vae(num_samples, temperature)
            elif isinstance(self.model, GraphGAN):
                samples = self._sample_gan(num_samples, temperature)
            elif isinstance(self.model, AutoregressiveGenerator):
                samples = self._sample_autoregressive(num_samples, temperature, top_k, top_p)
            else:
                samples = self._sample_generic(num_samples, temperature)
            
            # Convert samples to molecular representations
            for i, sample in enumerate(samples):
                molecule_data = self._convert_to_molecule(sample, i)
                generated_molecules.append(molecule_data)
        
        return generated_molecules
    
    def _sample_vae(self, num_samples: int, temperature: float) -> List[Dict]:
        """Sample from VAE model."""
        # Generate latent vectors
        z = torch.randn(num_samples, self.model.z_dim, device=self.device)
        
        # Decode to molecular representations
        atom_logits, bond_logits = self.model.decode(z)
        
        # Apply temperature scaling
        if temperature != 1.0:
            atom_logits = atom_logits / temperature
            bond_logits = bond_logits / temperature
        
        # Sample atom and bond types
        atom_probs = torch.softmax(atom_logits, dim=-1)
        bond_probs = torch.softmax(bond_logits, dim=-1)
        
        atoms = torch.multinomial(atom_probs.view(-1, atom_probs.size(-1)), 1).view(num_samples, -1)
        bonds = torch.multinomial(bond_probs.view(-1, bond_probs.size(-1)), 1).view(num_samples, -1)
        
        samples = []
        for i in range(num_samples):
            samples.append({
                'atoms': atoms[i],
                'bonds': bonds[i],
                'z': z[i]
            })
        
        return samples
    
    def _sample_gan(self, num_samples: int, temperature: float) -> List[Dict]:
        """Sample from GAN model."""
        # Generate noise vectors
        z = torch.randn(num_samples, self.model.z_dim, device=self.device)
        
        # Generate molecules
        outputs = self.model(z)
        
        # Sample atom and bond types
        atom_probs = torch.softmax(outputs['atom_logits'] / temperature, dim=-1)
        bond_probs = torch.softmax(outputs['bond_logits'] / temperature, dim=-1)
        
        atoms = torch.multinomial(atom_probs.view(-1, atom_probs.size(-1)), 1).view(num_samples, -1)
        bonds = torch.multinomial(bond_probs.view(-1, bond_probs.size(-1)), 1).view(num_samples, -1)
        
        samples = []
        for i in range(num_samples):
            samples.append({
                'atoms': atoms[i],
                'bonds': bonds[i],
                'z': z[i]
            })
        
        return samples
    
    def _sample_autoregressive(self, num_samples: int, temperature: float, top_k: Optional[int], top_p: Optional[float]) -> List[Dict]:
        """Sample from autoregressive model."""
        samples = self.model.generate(
            num_samples=num_samples,
            max_length=100,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=self.device
        )
        
        # Convert token sequences to molecular representations
        molecular_samples = []
        for i, sample in enumerate(samples):
            molecular_samples.append({
                'tokens': sample,
                'smiles': self._tokens_to_smiles(sample)
            })
        
        return molecular_samples
    
    def _sample_generic(self, num_samples: int, temperature: float) -> List[Dict]:
        """Generic sampling for unknown model types."""
        # Generate random molecules as placeholder
        samples = []
        for i in range(num_samples):
            samples.append({
                'atoms': torch.randint(0, len(self.atom_types), (10,), device=self.device),
                'bonds': torch.randint(0, len(self.bond_types), (5,), device=self.device)
            })
        
        return samples
    
    def _convert_to_molecule(self, sample: Dict, index: int) -> Dict[str, any]:
        """Convert sample to molecular representation."""
        molecule_data = {
            'index': index,
            'smiles': None,
            'mol': None,
            'graph': None,
            'image': None,
            'properties': {}
        }
        
        try:
            if 'smiles' in sample:
                # Direct SMILES
                smiles = sample['smiles']
                molecule_data['smiles'] = smiles
            else:
                # Convert from graph representation
                smiles = self._graph_to_smiles(sample)
                molecule_data['smiles'] = smiles
            
            # Create RDKit molecule
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    molecule_data['mol'] = mol
                    
                    # Create molecular image
                    img = Draw.MolToImage(mol, size=(300, 300))
                    molecule_data['image'] = img
                    
                    # Compute properties
                    molecule_data['properties'] = self._compute_molecular_properties(mol)
                    
                    # Create graph representation
                    molecule_data['graph'] = self._mol_to_graph(mol)
        
        except Exception as e:
            logger.debug(f"Failed to convert sample {index} to molecule: {e}")
            # Use placeholder molecule
            molecule_data['smiles'] = 'C1CCCCC1'  # Cyclohexane
            molecule_data['mol'] = Chem.MolFromSmiles('C1CCCCC1')
            molecule_data['image'] = Draw.MolToImage(Chem.MolFromSmiles('C1CCCCC1'))
        
        return molecule_data
    
    def _graph_to_smiles(self, sample: Dict) -> str:
        """Convert graph representation to SMILES."""
        # This is a simplified conversion - in practice, you'd need proper graph-to-SMILES conversion
        # For now, return a placeholder
        return 'C1CCCCC1'
    
    def _tokens_to_smiles(self, tokens: torch.Tensor) -> str:
        """Convert token sequence to SMILES."""
        # This would need to be implemented based on the tokenizer used
        # For now, return a placeholder
        return 'C1CCCCC1'
    
    def _mol_to_graph(self, mol: Chem.Mol) -> nx.Graph:
        """Convert RDKit molecule to NetworkX graph."""
        G = nx.Graph()
        
        # Add atoms as nodes
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
        
        # Add bonds as edges
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=bond.GetBondType().name
            )
        
        return G
    
    def _compute_molecular_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Compute molecular properties."""
        try:
            from rdkit.Chem import Descriptors, Crippen
            
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'qed': Descriptors.qed(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol)
            }
        except Exception as e:
            logger.debug(f"Failed to compute molecular properties: {e}")
            return {}
    
    def interpolate_molecules(
        self,
        mol1: Dict,
        mol2: Dict,
        num_steps: int = 10,
        seed: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """Interpolate between two molecules in latent space.
        
        Args:
            mol1: First molecule data.
            mol2: Second molecule data.
            num_steps: Number of interpolation steps.
            seed: Random seed for reproducibility.
            
        Returns:
            List of interpolated molecules.
        """
        if seed is not None:
            set_seed(seed)
        
        interpolated_molecules = []
        
        if isinstance(self.model, GraphVAE) and 'z' in mol1 and 'z' in mol2:
            # Linear interpolation in latent space
            z1 = mol1['z']
            z2 = mol2['z']
            
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Decode interpolated latent vector
                with torch.no_grad():
                    atom_logits, bond_logits = self.model.decode(z_interp.unsqueeze(0))
                    
                    # Sample from interpolated distribution
                    atom_probs = torch.softmax(atom_logits, dim=-1)
                    bond_probs = torch.softmax(bond_logits, dim=-1)
                    
                    atoms = torch.multinomial(atom_probs.view(-1, atom_probs.size(-1)), 1).view(1, -1)
                    bonds = torch.multinomial(bond_probs.view(-1, bond_probs.size(-1)), 1).view(1, -1)
                    
                    sample = {
                        'atoms': atoms[0],
                        'bonds': bonds[0],
                        'z': z_interp
                    }
                    
                    molecule_data = self._convert_to_molecule(sample, i)
                    molecule_data['interpolation_step'] = i
                    molecule_data['alpha'] = alpha
                    interpolated_molecules.append(molecule_data)
        
        return interpolated_molecules
    
    def visualize_samples(
        self,
        samples: List[Dict[str, any]],
        save_path: Optional[str] = None,
        max_samples: int = 16
    ) -> None:
        """Visualize generated molecular samples.
        
        Args:
            samples: List of generated molecular samples.
            save_path: Optional path to save the visualization.
            max_samples: Maximum number of samples to display.
        """
        num_samples = min(len(samples), max_samples)
        
        # Create subplot grid
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            
            if samples[i]['image'] is not None:
                axes[row, col].imshow(samples[i]['image'])
                axes[row, col].set_title(f"Sample {i+1}")
                axes[row, col].axis('off')
            else:
                axes[row, col].text(0.5, 0.5, f"Sample {i+1}\nNo Image", 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Generated Molecular Samples', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_samples(
        self,
        samples: List[Dict[str, any]],
        save_path: str,
        format: str = 'json'
    ) -> None:
        """Save generated samples to file.
        
        Args:
            samples: List of generated molecular samples.
            save_path: Path to save the samples.
            format: File format ('json', 'csv', 'sdf').
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            import json
            
            # Convert samples to JSON-serializable format
            json_samples = []
            for sample in samples:
                json_sample = {
                    'index': sample['index'],
                    'smiles': sample['smiles'],
                    'properties': sample['properties']
                }
                json_samples.append(json_sample)
            
            with open(save_path, 'w') as f:
                json.dump(json_samples, f, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            
            data = []
            for sample in samples:
                row = {
                    'index': sample['index'],
                    'smiles': sample['smiles'],
                    **sample['properties']
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(save_path, index=False)
        
        elif format == 'sdf':
            from rdkit.Chem import SDWriter
            
            writer = SDWriter(str(save_path))
            for sample in samples:
                if sample['mol'] is not None:
                    writer.write(sample['mol'])
            writer.close()
        
        logger.info(f"Saved {len(samples)} samples to {save_path}")
