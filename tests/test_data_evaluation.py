"""Unit tests for data handling and evaluation."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.data.dataset import load_zinc_dataset, generate_toy_dataset, create_data_loaders
from src.evaluation.metrics import MolecularMetrics
from src.utils.utils import save_checkpoint, load_checkpoint, format_time


class TestDataHandling:
    """Test data handling functions."""
    
    def test_generate_toy_dataset(self):
        """Test toy dataset generation."""
        dataset = generate_toy_dataset(size=100)
        
        assert len(dataset) == 100
        assert all(isinstance(smiles, str) for smiles in dataset)
        assert all(len(smiles) > 0 for smiles in dataset)
    
    def test_load_zinc_dataset_nonexistent(self):
        """Test loading ZINC dataset when file doesn't exist."""
        with patch('src.data.dataset.Path.exists', return_value=False):
            dataset = load_zinc_dataset("nonexistent_path.csv", subset_size=50)
            
            assert len(dataset) == 50
            assert all(isinstance(smiles, str) for smiles in dataset)
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        from src.data.dataset import MolecularDataset
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1", "CCCC", "CCCCC"]
        dataset = MolecularDataset(smiles_list)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset=dataset,
            batch_size=2,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Test that we can iterate through loaders
        for batch in train_loader:
            assert 'atoms' in batch
            assert 'bonds' in batch
            assert 'bond_types' in batch
            assert 'smiles' in batch
            break


class TestMolecularMetrics:
    """Test molecular evaluation metrics."""
    
    def test_qed_computation(self):
        """Test QED score computation."""
        metrics = MolecularMetrics()
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"]
        results = metrics.compute_qed(smiles_list)
        
        assert 'qed_mean' in results
        assert 'qed_std' in results
        assert 'qed_count' in results
        
        assert results['qed_count'] > 0
        assert 0 <= results['qed_mean'] <= 1
    
    def test_logp_computation(self):
        """Test LogP computation."""
        metrics = MolecularMetrics()
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"]
        results = metrics.compute_logp(smiles_list)
        
        assert 'logp_mean' in results
        assert 'logp_std' in results
        assert 'logp_count' in results
        
        assert results['logp_count'] > 0
    
    def test_molecular_weight_computation(self):
        """Test molecular weight computation."""
        metrics = MolecularMetrics()
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"]
        results = metrics.compute_molecular_weight(smiles_list)
        
        assert 'mw_mean' in results
        assert 'mw_std' in results
        assert 'mw_count' in results
        
        assert results['mw_count'] > 0
        assert results['mw_mean'] > 0
    
    def test_lipinski_compliance(self):
        """Test Lipinski's Rule of Five compliance."""
        metrics = MolecularMetrics()
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"]
        results = metrics.compute_lipinski_rule(smiles_list)
        
        assert 'lipinski_compliance' in results
        assert 'compliant_count' in results
        assert 'total_count' in results
        
        assert results['total_count'] > 0
        assert 0 <= results['lipinski_compliance'] <= 1
    
    def test_all_metrics_computation(self):
        """Test computation of all metrics."""
        metrics = MolecularMetrics()
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"]
        results = metrics.compute_all_metrics(smiles_list)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'validity', 'uniqueness', 'novelty',
            'qed_mean', 'sa_mean', 'logp_mean',
            'lipinski_compliance', 'mw_mean'
        ]
        
        for metric in expected_metrics:
            assert metric in results
    
    def test_metrics_report(self):
        """Test metrics report generation."""
        metrics = MolecularMetrics()
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"]
        report = metrics.create_metrics_report(smiles_list)
        
        assert 'summary' in report
        assert 'detailed_metrics' in report
        
        summary = report['summary']
        assert 'total_molecules' in summary
        assert 'validity_rate' in summary
        assert 'uniqueness_rate' in summary


class TestUtils:
    """Test utility functions."""
    
    def test_format_time(self):
        """Test time formatting."""
        assert format_time(30) == "30.0s"
        assert format_time(90) == "1.5m"
        assert format_time(7200) == "2.0h"
    
    def test_save_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        import tempfile
        import os
        
        # Create a simple model
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
            
            # Save checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=10,
                loss=0.5,
                path=checkpoint_path,
                extra_data={"test": "value"}
            )
            
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint
            loaded_checkpoint = load_checkpoint(
                model=model,
                optimizer=optimizer,
                path=checkpoint_path,
                device=torch.device('cpu')
            )
            
            assert loaded_checkpoint['epoch'] == 10
            assert loaded_checkpoint['loss'] == 0.5
            assert loaded_checkpoint['test'] == "value"


class TestDataLoaders:
    """Test data loader functionality."""
    
    def test_batch_processing(self):
        """Test batch processing in data loaders."""
        from src.data.dataset import MolecularDataset, collate_molecular_batch
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"]
        dataset = MolecularDataset(smiles_list)
        
        if len(dataset) > 0:
            # Create a batch
            batch = [dataset[i] for i in range(min(2, len(dataset)))]
            collated = collate_molecular_batch(batch)
            
            # Check batch structure
            assert 'atoms' in collated
            assert 'bonds' in collated
            assert 'bond_types' in collated
            assert 'num_atoms' in collated
            assert 'num_bonds' in collated
            assert 'smiles' in collated
            
            # Check tensor shapes
            assert collated['atoms'].dim() == 2
            assert collated['bonds'].dim() == 3
            assert collated['bond_types'].dim() == 2
            assert collated['num_atoms'].dim() == 1
            assert collated['num_bonds'].dim() == 1


class TestErrorHandling:
    """Test error handling in various components."""
    
    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES."""
        from src.data.dataset import MolecularDataset
        
        # Mix of valid and invalid SMILES
        smiles_list = ["C1CCCCC1", "invalid_smiles", "CCCCCC", "another_invalid"]
        dataset = MolecularDataset(smiles_list)
        
        # Should handle invalid SMILES gracefully
        assert len(dataset) > 0
        assert len(dataset) < len(smiles_list)  # Some should be filtered out
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        from src.data.dataset import MolecularDataset
        
        smiles_list = []
        dataset = MolecularDataset(smiles_list)
        
        assert len(dataset) == 0
    
    def test_metrics_with_empty_list(self):
        """Test metrics computation with empty lists."""
        metrics = MolecularMetrics()
        
        results = metrics.compute_validity([])
        assert results['validity'] == 0.0
        assert results['total_count'] == 0
        
        results = metrics.compute_uniqueness([])
        assert results['uniqueness'] == 0.0
        assert results['total_count'] == 0


class TestPerformance:
    """Test performance-related functionality."""
    
    def test_large_batch_handling(self):
        """Test handling of large batches."""
        from src.data.dataset import MolecularDataset, collate_molecular_batch
        
        # Create a larger dataset
        smiles_list = ["C1CCCCC1"] * 100  # Repeat same molecule
        dataset = MolecularDataset(smiles_list)
        
        if len(dataset) > 0:
            # Create a large batch
            batch_size = min(50, len(dataset))
            batch = [dataset[i] for i in range(batch_size)]
            
            # Should handle large batches efficiently
            collated = collate_molecular_batch(batch)
            
            assert collated['atoms'].shape[0] == batch_size
            assert collated['bonds'].shape[0] == batch_size
    
    def test_memory_efficiency(self):
        """Test memory efficiency of data processing."""
        from src.data.dataset import MolecularDataset
        
        # Create dataset with many molecules
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"] * 100
        dataset = MolecularDataset(smiles_list)
        
        # Should not consume excessive memory
        assert len(dataset) > 0
        
        # Test that we can iterate through dataset without memory issues
        count = 0
        for item in dataset:
            count += 1
            if count > 10:  # Limit to avoid long test times
                break
        
        assert count > 0


if __name__ == "__main__":
    pytest.main([__file__])
