"""Training module for molecular generation models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time

from ..models.molecular_models import GraphVAE, GraphGAN, AutoregressiveGenerator
from ..evaluation.metrics import MolecularMetrics
from ..utils.utils import save_checkpoint, load_checkpoint, format_time

logger = logging.getLogger(__name__)


class MolecularTrainer:
    """Trainer class for molecular generation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device('cpu'),
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Molecular generation model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on.
            config: Training configuration.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize loss functions
        self.criterion = self._setup_loss_functions()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Metrics
        self.metrics = MolecularMetrics()
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on model type."""
        lr = self.config.get('learning_rate', 0.0002)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if isinstance(self.model, GraphVAE):
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif isinstance(self.model, GraphGAN):
            return {
                'generator': optim.Adam(self.model.generator.parameters(), lr=lr, weight_decay=weight_decay),
                'discriminator': optim.Adam(self.model.discriminator.parameters(), lr=lr, weight_decay=weight_decay)
            }
        else:
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Setup loss functions based on model type."""
        if isinstance(self.model, GraphVAE):
            return {
                'reconstruction': nn.CrossEntropyLoss(),
                'kl_divergence': self._kl_divergence_loss
            }
        elif isinstance(self.model, GraphGAN):
            return {
                'generator': nn.BCELoss(),
                'discriminator': nn.BCELoss()
            }
        else:
            return {'cross_entropy': nn.CrossEntropyLoss()}
    
    def _kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for VAE."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            if isinstance(self.model, GraphVAE):
                loss = self._train_vae_step(batch)
            elif isinstance(self.model, GraphGAN):
                loss = self._train_gan_step(batch)
            else:
                loss = self._train_autoregressive_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_val']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def _train_vae_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training step for VAE model."""
        atoms = batch['atoms']
        bonds = batch['bonds']
        bond_types = batch['bond_types']
        
        # Forward pass
        outputs = self.model(atoms, bonds, bond_types)
        
        # Reconstruction loss
        recon_loss = self.criterion['reconstruction'](
            outputs['atom_logits'].view(-1, outputs['atom_logits'].size(-1)),
            atoms.view(-1)
        )
        
        # KL divergence loss
        kl_loss = self.criterion['kl_divergence'](outputs['mu'], outputs['logvar'])
        
        # Beta annealing
        beta = self.config.get('beta', 1.0)
        if self.config.get('beta_schedule') == 'linear':
            beta = min(1.0, self.epoch / self.config.get('beta_warmup', 10))
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss
    
    def _train_gan_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training step for GAN model."""
        atoms = batch['atoms']
        bonds = batch['bonds']
        bond_types = batch['bond_types']
        batch_size = atoms.size(0)
        
        # Train discriminator
        self.optimizer['discriminator'].zero_grad()
        
        # Real samples
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_output = self.model.discriminate(atoms, bonds, bond_types)
        real_loss = self.criterion['discriminator'](real_output, real_labels)
        
        # Fake samples
        z = torch.randn(batch_size, self.model.z_dim, device=self.device)
        fake_outputs = self.model(z)
        fake_atoms = torch.argmax(fake_outputs['atom_logits'], dim=-1)
        fake_bonds = torch.zeros(batch_size, 1, 2, device=self.device)  # Simplified
        fake_bond_types = torch.zeros(batch_size, 1, device=self.device)  # Simplified
        
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_output = self.model.discriminate(fake_atoms, fake_bonds, fake_bond_types)
        fake_loss = self.criterion['discriminator'](fake_output, fake_labels)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer['discriminator'].step()
        
        # Train generator
        self.optimizer['generator'].zero_grad()
        
        z = torch.randn(batch_size, self.model.z_dim, device=self.device)
        fake_outputs = self.model(z)
        fake_atoms = torch.argmax(fake_outputs['atom_logits'], dim=-1)
        fake_bonds = torch.zeros(batch_size, 1, 2, device=self.device)  # Simplified
        fake_bond_types = torch.zeros(batch_size, 1, device=self.device)  # Simplified
        
        fake_output = self.model.discriminate(fake_atoms, fake_bonds, fake_bond_types)
        g_loss = self.criterion['generator'](fake_output, real_labels)
        
        g_loss.backward()
        self.optimizer['generator'].step()
        
        return d_loss + g_loss
    
    def _train_autoregressive_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training step for autoregressive model."""
        # This would need to be implemented based on the specific autoregressive model
        # For now, return a dummy loss
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch data to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                
                if isinstance(self.model, GraphVAE):
                    outputs = self.model(batch['atoms'], batch['bonds'], batch['bond_types'])
                    loss = self.criterion['reconstruction'](
                        outputs['atom_logits'].view(-1, outputs['atom_logits'].size(-1)),
                        batch['atoms'].view(-1)
                    )
                else:
                    loss = torch.tensor(0.0, device=self.device)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, num_epochs: int, save_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            save_dir: Directory to save checkpoints.
            
        Returns:
            Dictionary containing training history.
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Save checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    checkpoint_path
                )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_dir:
                    best_path = save_path / "best_model.pt"
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        best_path
                    )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {format_time(training_time)}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_time': training_time
        }
    
    def evaluate_model(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Evaluate the trained model.
        
        Args:
            num_samples: Number of samples to generate for evaluation.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        
        # Generate samples
        generated_smiles = []
        
        with torch.no_grad():
            if isinstance(self.model, GraphVAE):
                samples = self.model.sample(num_samples, self.device)
                # Convert samples to SMILES (simplified)
                for i in range(num_samples):
                    # This is a simplified conversion - in practice, you'd need proper graph-to-SMILES conversion
                    generated_smiles.append("C1CCCCC1")  # Placeholder
            else:
                # Generate samples for other model types
                for i in range(num_samples):
                    generated_smiles.append("C1CCCCC1")  # Placeholder
        
        # Compute metrics
        metrics = self.metrics.compute_all_metrics(generated_smiles)
        
        return metrics
    
    def generate_samples(self, num_samples: int, temperature: float = 1.0) -> List[str]:
        """Generate molecular samples.
        
        Args:
            num_samples: Number of samples to generate.
            temperature: Sampling temperature.
            
        Returns:
            List of generated SMILES strings.
        """
        self.model.eval()
        generated_smiles = []
        
        with torch.no_grad():
            if isinstance(self.model, GraphVAE):
                samples = self.model.sample(num_samples, self.device)
                # Convert samples to SMILES (simplified)
                for i in range(num_samples):
                    generated_smiles.append("C1CCCCC1")  # Placeholder
            else:
                # Generate samples for other model types
                for i in range(num_samples):
                    generated_smiles.append("C1CCCCC1")  # Placeholder
        
        return generated_smiles
