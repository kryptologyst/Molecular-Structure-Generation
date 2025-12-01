"""Evaluation metrics for molecular generation."""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MolecularMetrics:
    """Collection of molecular evaluation metrics."""
    
    def __init__(self, reference_smiles: Optional[List[str]] = None):
        """Initialize molecular metrics.
        
        Args:
            reference_smiles: Reference SMILES for novelty calculation.
        """
        self.reference_smiles = reference_smiles or []
        self.reference_mols = [Chem.MolFromSmiles(smi) for smi in self.reference_smiles if Chem.MolFromSmiles(smi) is not None]
    
    def compute_validity(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute molecular validity metrics.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            
        Returns:
            Dictionary containing validity metrics.
        """
        valid_mols = []
        invalid_count = 0
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
            else:
                invalid_count += 1
        
        validity = len(valid_mols) / len(smiles_list) if smiles_list else 0.0
        
        return {
            'validity': validity,
            'valid_count': len(valid_mols),
            'invalid_count': invalid_count,
            'total_count': len(smiles_list)
        }
    
    def compute_uniqueness(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute molecular uniqueness metrics.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            
        Returns:
            Dictionary containing uniqueness metrics.
        """
        if not smiles_list:
            return {'uniqueness': 0.0, 'unique_count': 0, 'total_count': 0}
        
        # Canonicalize SMILES
        canonical_smiles = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles.append(Chem.MolToSmiles(mol))
        
        unique_smiles = set(canonical_smiles)
        uniqueness = len(unique_smiles) / len(canonical_smiles) if canonical_smiles else 0.0
        
        return {
            'uniqueness': uniqueness,
            'unique_count': len(unique_smiles),
            'total_count': len(canonical_smiles)
        }
    
    def compute_novelty(self, generated_smiles: List[str]) -> Dict[str, float]:
        """Compute molecular novelty metrics.
        
        Args:
            generated_smiles: List of generated SMILES strings.
            
        Returns:
            Dictionary containing novelty metrics.
        """
        if not generated_smiles or not self.reference_smiles:
            return {'novelty': 0.0, 'novel_count': 0, 'total_count': 0}
        
        # Canonicalize generated SMILES
        generated_canonical = set()
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                generated_canonical.add(Chem.MolToSmiles(mol))
        
        # Canonicalize reference SMILES
        reference_canonical = set()
        for smiles in self.reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                reference_canonical.add(Chem.MolToSmiles(mol))
        
        # Count novel molecules
        novel_molecules = generated_canonical - reference_canonical
        novelty = len(novel_molecules) / len(generated_canonical) if generated_canonical else 0.0
        
        return {
            'novelty': novelty,
            'novel_count': len(novel_molecules),
            'total_count': len(generated_canonical)
        }
    
    def compute_qed(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute QED (Quantitative Estimate of Drug-likeness) scores.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            
        Returns:
            Dictionary containing QED statistics.
        """
        qed_scores = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    qed_score = Descriptors.qed(mol)
                    qed_scores.append(qed_score)
                except Exception as e:
                    logger.debug(f"Failed to compute QED for {smiles}: {e}")
        
        if not qed_scores:
            return {'qed_mean': 0.0, 'qed_std': 0.0, 'qed_count': 0}
        
        return {
            'qed_mean': np.mean(qed_scores),
            'qed_std': np.std(qed_scores),
            'qed_count': len(qed_scores),
            'qed_scores': qed_scores
        }
    
    def compute_sa_score(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute SA (Synthetic Accessibility) scores.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            
        Returns:
            Dictionary containing SA score statistics.
        """
        sa_scores = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    # Use BertzCT as a proxy for SA score
                    sa_score = Descriptors.BertzCT(mol)
                    sa_scores.append(sa_score)
                except Exception as e:
                    logger.debug(f"Failed to compute SA score for {smiles}: {e}")
        
        if not sa_scores:
            return {'sa_mean': 0.0, 'sa_std': 0.0, 'sa_count': 0}
        
        return {
            'sa_mean': np.mean(sa_scores),
            'sa_std': np.std(sa_scores),
            'sa_count': len(sa_scores),
            'sa_scores': sa_scores
        }
    
    def compute_logp(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute LogP (partition coefficient) scores.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            
        Returns:
            Dictionary containing LogP statistics.
        """
        logp_scores = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    logp_score = Crippen.MolLogP(mol)
                    logp_scores.append(logp_score)
                except Exception as e:
                    logger.debug(f"Failed to compute LogP for {smiles}: {e}")
        
        if not logp_scores:
            return {'logp_mean': 0.0, 'logp_std': 0.0, 'logp_count': 0}
        
        return {
            'logp_mean': np.mean(logp_scores),
            'logp_std': np.std(logp_scores),
            'logp_count': len(logp_scores),
            'logp_scores': logp_scores
        }
    
    def compute_lipinski_rule(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute Lipinski's Rule of Five compliance.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            
        Returns:
            Dictionary containing Lipinski compliance statistics.
        """
        compliant_count = 0
        total_count = 0
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    # Check Lipinski's Rule of Five
                    mw = Descriptors.MolWt(mol)
                    logp = Crippen.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    
                    # Rule of Five: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10
                    if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                        compliant_count += 1
                    
                    total_count += 1
                except Exception as e:
                    logger.debug(f"Failed to compute Lipinski compliance for {smiles}: {e}")
        
        compliance_rate = compliant_count / total_count if total_count > 0 else 0.0
        
        return {
            'lipinski_compliance': compliance_rate,
            'compliant_count': compliant_count,
            'total_count': total_count
        }
    
    def compute_molecular_weight(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute molecular weight statistics.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            
        Returns:
            Dictionary containing molecular weight statistics.
        """
        mw_scores = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    mw_score = Descriptors.MolWt(mol)
                    mw_scores.append(mw_score)
                except Exception as e:
                    logger.debug(f"Failed to compute molecular weight for {smiles}: {e}")
        
        if not mw_scores:
            return {'mw_mean': 0.0, 'mw_std': 0.0, 'mw_count': 0}
        
        return {
            'mw_mean': np.mean(mw_scores),
            'mw_std': np.std(mw_scores),
            'mw_count': len(mw_scores),
            'mw_scores': mw_scores
        }
    
    def compute_all_metrics(self, smiles_list: List[str]) -> Dict[str, Union[float, Dict]]:
        """Compute all available metrics.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            
        Returns:
            Dictionary containing all computed metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self.compute_validity(smiles_list))
        metrics.update(self.compute_uniqueness(smiles_list))
        metrics.update(self.compute_novelty(smiles_list))
        
        # Property-based metrics
        metrics.update(self.compute_qed(smiles_list))
        metrics.update(self.compute_sa_score(smiles_list))
        metrics.update(self.compute_logp(smiles_list))
        metrics.update(self.compute_lipinski_rule(smiles_list))
        metrics.update(self.compute_molecular_weight(smiles_list))
        
        return metrics
    
    def create_metrics_report(self, smiles_list: List[str], save_path: Optional[str] = None) -> Dict[str, Union[float, Dict]]:
        """Create a comprehensive metrics report.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            save_path: Optional path to save the report.
            
        Returns:
            Dictionary containing the metrics report.
        """
        metrics = self.compute_all_metrics(smiles_list)
        
        # Create summary
        report = {
            'summary': {
                'total_molecules': len(smiles_list),
                'valid_molecules': metrics.get('valid_count', 0),
                'unique_molecules': metrics.get('unique_count', 0),
                'novel_molecules': metrics.get('novel_count', 0),
                'validity_rate': metrics.get('validity', 0.0),
                'uniqueness_rate': metrics.get('uniqueness', 0.0),
                'novelty_rate': metrics.get('novelty', 0.0),
                'lipinski_compliance': metrics.get('lipinski_compliance', 0.0)
            },
            'detailed_metrics': metrics
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_metrics_distribution(self, smiles_list: List[str], save_path: Optional[str] = None) -> None:
        """Plot distribution of molecular metrics.
        
        Args:
            smiles_list: List of SMILES strings to evaluate.
            save_path: Optional path to save the plot.
        """
        metrics = self.compute_all_metrics(smiles_list)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Molecular Metrics Distribution', fontsize=16)
        
        # QED distribution
        if 'qed_scores' in metrics and metrics['qed_scores']:
            axes[0, 0].hist(metrics['qed_scores'], bins=30, alpha=0.7, color='blue')
            axes[0, 0].set_title('QED Distribution')
            axes[0, 0].set_xlabel('QED Score')
            axes[0, 0].set_ylabel('Frequency')
        
        # SA Score distribution
        if 'sa_scores' in metrics and metrics['sa_scores']:
            axes[0, 1].hist(metrics['sa_scores'], bins=30, alpha=0.7, color='green')
            axes[0, 1].set_title('SA Score Distribution')
            axes[0, 1].set_xlabel('SA Score')
            axes[0, 1].set_ylabel('Frequency')
        
        # LogP distribution
        if 'logp_scores' in metrics and metrics['logp_scores']:
            axes[0, 2].hist(metrics['logp_scores'], bins=30, alpha=0.7, color='red')
            axes[0, 2].set_title('LogP Distribution')
            axes[0, 2].set_xlabel('LogP')
            axes[0, 2].set_ylabel('Frequency')
        
        # Molecular Weight distribution
        if 'mw_scores' in metrics and metrics['mw_scores']:
            axes[1, 0].hist(metrics['mw_scores'], bins=30, alpha=0.7, color='purple')
            axes[1, 0].set_title('Molecular Weight Distribution')
            axes[1, 0].set_xlabel('Molecular Weight')
            axes[1, 0].set_ylabel('Frequency')
        
        # Validity metrics pie chart
        valid_count = metrics.get('valid_count', 0)
        invalid_count = metrics.get('invalid_count', 0)
        if valid_count + invalid_count > 0:
            axes[1, 1].pie([valid_count, invalid_count], labels=['Valid', 'Invalid'], 
                          colors=['green', 'red'], autopct='%1.1f%%')
            axes[1, 1].set_title('Validity Distribution')
        
        # Summary metrics bar chart
        summary_metrics = ['validity', 'uniqueness', 'novelty', 'lipinski_compliance']
        summary_values = [metrics.get(metric, 0.0) for metric in summary_metrics]
        axes[1, 2].bar(summary_metrics, summary_values, color=['blue', 'green', 'red', 'purple'])
        axes[1, 2].set_title('Summary Metrics')
        axes[1, 2].set_ylabel('Rate')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
