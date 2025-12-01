"""Molecular generation models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from rdkit import Chem
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class GraphVAE(nn.Module):
    """Graph Variational Autoencoder for molecular generation."""
    
    def __init__(
        self,
        atom_types: int = 10,
        bond_types: int = 4,
        max_atoms: int = 50,
        hidden_dim: int = 128,
        z_dim: int = 100,
        dropout: float = 0.1
    ):
        """Initialize GraphVAE.
        
        Args:
            atom_types: Number of atom types.
            bond_types: Number of bond types.
            max_atoms: Maximum number of atoms.
            hidden_dim: Hidden dimension size.
            z_dim: Latent dimension size.
            dropout: Dropout rate.
        """
        super(GraphVAE, self).__init__()
        
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        # Encoder
        self.atom_encoder = nn.Embedding(atom_types, hidden_dim)
        self.bond_encoder = nn.Embedding(bond_types, hidden_dim)
        
        self.encoder_conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.encoder_conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.encoder_fc = nn.Linear(hidden_dim * max_atoms, hidden_dim * 2)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(z_dim, hidden_dim * max_atoms)
        self.decoder_conv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, padding=1)
        self.decoder_conv2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.atom_decoder = nn.Linear(hidden_dim, atom_types)
        self.bond_decoder = nn.Linear(hidden_dim, bond_types)
        
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, atoms: torch.Tensor, bonds: torch.Tensor, bond_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode molecular graph to latent space.
        
        Args:
            atoms: Atom types tensor [batch_size, max_atoms].
            bonds: Bond indices tensor [batch_size, max_bonds, 2].
            bond_types: Bond types tensor [batch_size, max_bonds].
            
        Returns:
            Tuple of (mu, logvar) for latent space.
        """
        batch_size = atoms.size(0)
        
        # Encode atoms
        atom_emb = self.atom_encoder(atoms)  # [batch_size, max_atoms, hidden_dim]
        atom_emb = atom_emb.transpose(1, 2)  # [batch_size, hidden_dim, max_atoms]
        
        # Apply convolutions
        x = F.relu(self.encoder_conv1(atom_emb))
        x = self.dropout(x)
        x = F.relu(self.encoder_conv2(x))
        x = self.dropout(x)
        
        # Flatten and encode
        x = x.transpose(1, 2).contiguous().view(batch_size, -1)
        x = F.relu(self.encoder_fc(x))
        
        # Get latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to molecular graph.
        
        Args:
            z: Latent vector [batch_size, z_dim].
            
        Returns:
            Tuple of (atom_logits, bond_logits).
        """
        batch_size = z.size(0)
        
        # Decode to hidden representation
        x = F.relu(self.decoder_fc(z))
        x = x.view(batch_size, self.hidden_dim, self.max_atoms)
        
        # Apply transpose convolutions
        x = F.relu(self.decoder_conv1(x))
        x = self.dropout(x)
        x = F.relu(self.decoder_conv2(x))
        x = self.dropout(x)
        
        # Transpose back to [batch_size, max_atoms, hidden_dim]
        x = x.transpose(1, 2)
        
        # Decode to atom and bond types
        atom_logits = self.atom_decoder(x)
        bond_logits = self.bond_decoder(x)
        
        return atom_logits, bond_logits
    
    def forward(self, atoms: torch.Tensor, bonds: torch.Tensor, bond_types: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            atoms: Atom types tensor.
            bonds: Bond indices tensor.
            bond_types: Bond types tensor.
            
        Returns:
            Dictionary containing outputs and latent parameters.
        """
        # Encode
        mu, logvar = self.encode(atoms, bonds, bond_types)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        atom_logits, bond_logits = self.decode(z)
        
        return {
            'atom_logits': atom_logits,
            'bond_logits': bond_logits,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def sample(self, num_samples: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample molecules from the model.
        
        Args:
            num_samples: Number of samples to generate.
            device: Device to generate on.
            
        Returns:
            Dictionary containing generated molecules.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim, device=device)
            atom_logits, bond_logits = self.decode(z)
            
            return {
                'atom_logits': atom_logits,
                'bond_logits': bond_logits,
                'z': z
            }


class GraphGAN(nn.Module):
    """Graph Generative Adversarial Network for molecular generation."""
    
    def __init__(
        self,
        atom_types: int = 10,
        bond_types: int = 4,
        max_atoms: int = 50,
        hidden_dim: int = 128,
        z_dim: int = 100
    ):
        """Initialize GraphGAN.
        
        Args:
            atom_types: Number of atom types.
            bond_types: Number of bond types.
            max_atoms: Maximum number of atoms.
            hidden_dim: Hidden dimension size.
            z_dim: Latent dimension size.
        """
        super(GraphGAN, self).__init__()
        
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        # Generator
        self.generator = GraphGenerator(
            atom_types=atom_types,
            bond_types=bond_types,
            max_atoms=max_atoms,
            hidden_dim=hidden_dim,
            z_dim=z_dim
        )
        
        # Discriminator
        self.discriminator = GraphDiscriminator(
            atom_types=atom_types,
            bond_types=bond_types,
            max_atoms=max_atoms,
            hidden_dim=hidden_dim
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate molecules from noise.
        
        Args:
            z: Noise vector [batch_size, z_dim].
            
        Returns:
            Dictionary containing generated molecules.
        """
        return self.generator(z)
    
    def discriminate(self, atoms: torch.Tensor, bonds: torch.Tensor, bond_types: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake molecules.
        
        Args:
            atoms: Atom types tensor.
            bonds: Bond indices tensor.
            bond_types: Bond types tensor.
            
        Returns:
            Discriminator output.
        """
        return self.discriminator(atoms, bonds, bond_types)


class GraphGenerator(nn.Module):
    """Generator network for GraphGAN."""
    
    def __init__(
        self,
        atom_types: int = 10,
        bond_types: int = 4,
        max_atoms: int = 50,
        hidden_dim: int = 128,
        z_dim: int = 100
    ):
        super(GraphGenerator, self).__init__()
        
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        # Generator layers
        self.fc1 = nn.Linear(z_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.atom_head = nn.Linear(hidden_dim, atom_types)
        self.bond_head = nn.Linear(hidden_dim, bond_types)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate molecular graphs from noise.
        
        Args:
            z: Noise vector [batch_size, z_dim].
            
        Returns:
            Dictionary containing generated molecules.
        """
        batch_size = z.size(0)
        
        # Generate features
        x = F.relu(self.fc1(z))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Generate atom and bond types
        atom_logits = self.atom_head(x)
        bond_logits = self.bond_head(x)
        
        return {
            'atom_logits': atom_logits,
            'bond_logits': bond_logits
        }


class GraphDiscriminator(nn.Module):
    """Discriminator network for GraphGAN."""
    
    def __init__(
        self,
        atom_types: int = 10,
        bond_types: int = 4,
        max_atoms: int = 50,
        hidden_dim: int = 128
    ):
        super(GraphDiscriminator, self).__init__()
        
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim
        
        # Embedding layers
        self.atom_embedding = nn.Embedding(atom_types, hidden_dim)
        self.bond_embedding = nn.Embedding(bond_types, hidden_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim * max_atoms, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, atoms: torch.Tensor, bonds: torch.Tensor, bond_types: torch.Tensor) -> torch.Tensor:
        """Discriminate molecular graphs.
        
        Args:
            atoms: Atom types tensor [batch_size, max_atoms].
            bonds: Bond indices tensor [batch_size, max_bonds, 2].
            bond_types: Bond types tensor [batch_size, max_bonds].
            
        Returns:
            Discriminator output [batch_size, 1].
        """
        batch_size = atoms.size(0)
        
        # Embed atoms
        atom_emb = self.atom_embedding(atoms)  # [batch_size, max_atoms, hidden_dim]
        atom_emb = atom_emb.transpose(1, 2)  # [batch_size, hidden_dim, max_atoms]
        
        # Apply convolutions
        x = F.relu(self.conv1(atom_emb))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        
        # Flatten and classify
        x = x.transpose(1, 2).contiguous().view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


class AutoregressiveGenerator(nn.Module):
    """Autoregressive molecular generator using transformer architecture."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_length: int = 100,
        dropout: float = 0.1
    ):
        """Initialize autoregressive generator.
        
        Args:
            vocab_size: Vocabulary size for SMILES tokens.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            max_length: Maximum sequence length.
            dropout: Dropout rate.
        """
        super(AutoregressiveGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for autoregressive generation.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size].
        """
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Transformer forward
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        output = self.transformer(x, x, tgt_mask=causal_mask)
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Output projection
        logits = self.output_projection(output)
        
        return logits
    
    def generate(
        self,
        num_samples: int,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Generate molecules autoregressively.
        
        Args:
            num_samples: Number of samples to generate.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            device: Device to generate on.
            
        Returns:
            Generated token sequences [num_samples, max_length].
        """
        self.eval()
        with torch.no_grad():
            # Start with BOS token (assuming 0 is BOS)
            generated = torch.zeros(num_samples, 1, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                # Get logits for next token
                logits = self.forward(generated)[:, -1, :]  # [num_samples, vocab_size]
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply nucleus sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences have EOS token (assuming 1 is EOS)
                if (next_token == 1).all():
                    break
            
            return generated
