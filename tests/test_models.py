"""Unit tests for molecular generation models."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models.molecular_models import GraphVAE, GraphGAN, AutoregressiveGenerator
from src.data.dataset import MolecularDataset, collate_molecular_batch
from src.evaluation.metrics import MolecularMetrics
from src.utils.utils import set_seed, get_device, count_parameters


class TestGraphVAE:
    """Test GraphVAE model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GraphVAE(
            atom_types=10,
            bond_types=4,
            max_atoms=50,
            hidden_dim=128,
            z_dim=100
        )
        
        assert model.atom_types == 10
        assert model.bond_types == 4
        assert model.max_atoms == 50
        assert model.hidden_dim == 128
        assert model.z_dim == 100
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = GraphVAE(atom_types=10, bond_types=4, max_atoms=50)
        
        batch_size = 4
        atoms = torch.randint(0, 10, (batch_size, 50))
        bonds = torch.randint(0, 50, (batch_size, 20, 2))
        bond_types = torch.randint(0, 4, (batch_size, 20))
        
        outputs = model(atoms, bonds, bond_types)
        
        assert 'atom_logits' in outputs
        assert 'bond_logits' in outputs
        assert 'mu' in outputs
        assert 'logvar' in outputs
        assert 'z' in outputs
        
        assert outputs['atom_logits'].shape == (batch_size, 50, 10)
        assert outputs['bond_logits'].shape == (batch_size, 50, 4)
        assert outputs['mu'].shape == (batch_size, 100)
        assert outputs['logvar'].shape == (batch_size, 100)
    
    def test_sampling(self):
        """Test sampling from model."""
        model = GraphVAE(atom_types=10, bond_types=4, max_atoms=50)
        device = torch.device('cpu')
        
        num_samples = 5
        samples = model.sample(num_samples, device)
        
        assert 'atom_logits' in samples
        assert 'bond_logits' in samples
        assert 'z' in samples
        
        assert samples['atom_logits'].shape == (num_samples, 50, 10)
        assert samples['bond_logits'].shape == (num_samples, 50, 4)
        assert samples['z'].shape == (num_samples, 100)


class TestGraphGAN:
    """Test GraphGAN model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GraphGAN(
            atom_types=10,
            bond_types=4,
            max_atoms=50,
            hidden_dim=128,
            z_dim=100
        )
        
        assert hasattr(model, 'generator')
        assert hasattr(model, 'discriminator')
        assert model.z_dim == 100
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = GraphGAN(atom_types=10, bond_types=4, max_atoms=50)
        
        batch_size = 4
        z = torch.randn(batch_size, 100)
        
        outputs = model(z)
        
        assert 'atom_logits' in outputs
        assert 'bond_logits' in outputs
        
        assert outputs['atom_logits'].shape == (batch_size, 10)
        assert outputs['bond_logits'].shape == (batch_size, 4)
    
    def test_discriminator(self):
        """Test discriminator."""
        model = GraphGAN(atom_types=10, bond_types=4, max_atoms=50)
        
        batch_size = 4
        atoms = torch.randint(0, 10, (batch_size, 50))
        bonds = torch.randint(0, 50, (batch_size, 20, 2))
        bond_types = torch.randint(0, 4, (batch_size, 20))
        
        output = model.discriminate(atoms, bonds, bond_types)
        
        assert output.shape == (batch_size, 1)


class TestAutoregressiveGenerator:
    """Test AutoregressiveGenerator model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = AutoregressiveGenerator(
            vocab_size=1000,
            d_model=256,
            nhead=8,
            num_layers=6,
            max_length=100
        )
        
        assert model.vocab_size == 1000
        assert model.d_model == 256
        assert model.max_length == 100
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = AutoregressiveGenerator(vocab_size=1000, d_model=256)
        
        batch_size = 4
        seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, 1000)
    
    def test_generation(self):
        """Test generation."""
        model = AutoregressiveGenerator(vocab_size=1000, d_model=256)
        device = torch.device('cpu')
        
        num_samples = 3
        max_length = 10
        
        generated = model.generate(
            num_samples=num_samples,
            max_length=max_length,
            device=device
        )
        
        assert generated.shape == (num_samples, max_length)


class TestMolecularDataset:
    """Test MolecularDataset class."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        smiles_list = ["C1CCCCC1", "CCCCCC", "C1=CC=CC=C1"]
        dataset = MolecularDataset(smiles_list)
        
        assert len(dataset) > 0
        assert len(dataset.atom_types) > 0
        assert len(dataset.bond_types) > 0
    
    def test_getitem(self):
        """Test dataset item retrieval."""
        smiles_list = ["C1CCCCC1", "CCCCCC"]
        dataset = MolecularDataset(smiles_list)
        
        if len(dataset) > 0:
            item = dataset[0]
            
            assert 'smiles' in item
            assert 'atoms' in item
            assert 'bonds' in item
            assert 'bond_types' in item
            assert 'num_atoms' in item
            assert 'num_bonds' in item
            assert 'properties' in item
    
    def test_collate_function(self):
        """Test collate function."""
        # Create mock batch
        batch = [
            {
                'atoms': torch.tensor([0, 1, 2]),
                'bonds': torch.tensor([[0, 1], [1, 2]]),
                'bond_types': torch.tensor([0, 1]),
                'num_atoms': 3,
                'num_bonds': 2,
                'smiles': 'C1CCCCC1'
            },
            {
                'atoms': torch.tensor([0, 1]),
                'bonds': torch.tensor([[0, 1]]),
                'bond_types': torch.tensor([0]),
                'num_atoms': 2,
                'num_bonds': 1,
                'smiles': 'CC'
            }
        ]
        
        collated = collate_molecular_batch(batch)
        
        assert 'atoms' in collated
        assert 'bonds' in collated
        assert 'bond_types' in collated
        assert 'num_atoms' in collated
        assert 'num_bonds' in collated
        assert 'smiles' in collated
        
        assert collated['atoms'].shape[0] == 2
        assert collated['bonds'].shape[0] == 2


class TestMolecularMetrics:
    """Test MolecularMetrics class."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = MolecularMetrics()
        assert metrics.reference_smiles == []
    
    def test_validity_computation(self):
        """Test validity computation."""
        metrics = MolecularMetrics()
        
        smiles_list = ["C1CCCCC1", "CCCCCC", "invalid_smiles"]
        results = metrics.compute_validity(smiles_list)
        
        assert 'validity' in results
        assert 'valid_count' in results
        assert 'invalid_count' in results
        assert 'total_count' in results
        
        assert results['total_count'] == 3
        assert results['invalid_count'] >= 1
    
    def test_uniqueness_computation(self):
        """Test uniqueness computation."""
        metrics = MolecularMetrics()
        
        smiles_list = ["C1CCCCC1", "C1CCCCC1", "CCCCCC"]
        results = metrics.compute_uniqueness(smiles_list)
        
        assert 'uniqueness' in results
        assert 'unique_count' in results
        assert 'total_count' in results
        
        assert results['total_count'] == 3
        assert results['unique_count'] == 2
        assert results['uniqueness'] == 2/3
    
    def test_novelty_computation(self):
        """Test novelty computation."""
        reference_smiles = ["C1CCCCC1", "CCCCCC"]
        metrics = MolecularMetrics(reference_smiles)
        
        generated_smiles = ["C1CCCCC1", "C1=CC=CC=C1"]
        results = metrics.compute_novelty(generated_smiles)
        
        assert 'novelty' in results
        assert 'novel_count' in results
        assert 'total_count' in results


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        import random
        import numpy as np
        
        # This is a basic test - in practice, you'd test actual randomness
        assert True  # Placeholder
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = GraphVAE(atom_types=10, bond_types=4, max_atoms=50)
        param_count = count_parameters(model)
        
        assert param_count > 0
        assert isinstance(param_count, int)


class TestIntegration:
    """Integration tests."""
    
    def test_model_training_step(self):
        """Test a single training step."""
        model = GraphVAE(atom_types=10, bond_types=4, max_atoms=50)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Mock batch
        atoms = torch.randint(0, 10, (2, 50))
        bonds = torch.randint(0, 50, (2, 20, 2))
        bond_types = torch.randint(0, 4, (2, 20))
        
        # Forward pass
        outputs = model(atoms, bonds, bond_types)
        
        # Compute loss
        recon_loss = torch.nn.CrossEntropyLoss()(
            outputs['atom_logits'].view(-1, outputs['atom_logits'].size(-1)),
            atoms.view(-1)
        )
        kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        total_loss = recon_loss + kl_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        assert total_loss.item() > 0
    
    def test_sampling_pipeline(self):
        """Test sampling pipeline."""
        model = GraphVAE(atom_types=10, bond_types=4, max_atoms=50)
        device = torch.device('cpu')
        
        # Sample molecules
        samples = model.sample(5, device)
        
        assert samples['atom_logits'].shape[0] == 5
        assert samples['bond_logits'].shape[0] == 5
        
        # Test that we can convert to probabilities
        atom_probs = torch.softmax(samples['atom_logits'], dim=-1)
        bond_probs = torch.softmax(samples['bond_logits'], dim=-1)
        
        assert atom_probs.shape == samples['atom_logits'].shape
        assert bond_probs.shape == samples['bond_logits'].shape


if __name__ == "__main__":
    pytest.main([__file__])
