"""Streamlit demo for molecular structure generation."""

import streamlit as st
import torch
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from pathlib import Path
import logging

# Import our modules
from src.models.molecular_models import GraphVAE, GraphGAN, AutoregressiveGenerator
from src.sampling.sampler import MolecularSampler
from src.evaluation.metrics import MolecularMetrics
from src.utils.utils import get_device, set_seed
from src.data.dataset import generate_toy_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Molecular Structure Generation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sample-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'sampler' not in st.session_state:
    st.session_state.sampler = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

def load_model(model_type: str, device: torch.device) -> torch.nn.Module:
    """Load a molecular generation model."""
    if model_type == "GraphVAE":
        model = GraphVAE(
            atom_types=10,
            bond_types=4,
            max_atoms=50,
            hidden_dim=128,
            z_dim=100
        )
    elif model_type == "GraphGAN":
        model = GraphGAN(
            atom_types=10,
            bond_types=4,
            max_atoms=50,
            hidden_dim=128,
            z_dim=100
        )
    elif model_type == "Autoregressive":
        model = AutoregressiveGenerator(
            vocab_size=1000,
            d_model=256,
            nhead=8,
            num_layers=6,
            max_length=100
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    return model

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧪 Molecular Structure Generation</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["GraphVAE", "GraphGAN", "Autoregressive"],
            help="Choose the type of molecular generation model"
        )
        
        # Device selection
        device_option = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda", "mps"],
            help="Select the device for computation"
        )
        device = get_device(device_option)
        
        # Sampling parameters
        st.subheader("Sampling Parameters")
        num_samples = st.slider("Number of Samples", 1, 50, 10)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=10000)
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            top_k = st.number_input("Top-k", value=None, min_value=1, max_value=100, help="Top-k sampling")
            top_p = st.slider("Top-p (Nucleus)", 0.0, 1.0, 1.0, 0.1, help="Nucleus sampling")
            
            if top_k == 0:
                top_k = None
            if top_p == 1.0:
                top_p = None
        
        # Load model button
        if st.button("Load Model", type="primary"):
            with st.spinner("Loading model..."):
                try:
                    st.session_state.model = load_model(model_type, device)
                    st.session_state.sampler = MolecularSampler(
                        st.session_state.model,
                        device
                    )
                    st.session_state.metrics = MolecularMetrics()
                    st.success(f"Model loaded successfully on {device}")
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
    
    # Main content
    if st.session_state.model is None:
        st.info("👈 Please load a model from the sidebar to get started")
        
        # Show model information
        st.subheader("Available Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **GraphVAE**
            - Variational Autoencoder for molecular graphs
            - Learns latent representations of molecules
            - Good for interpolation and controlled generation
            """)
        
        with col2:
            st.markdown("""
            **GraphGAN**
            - Generative Adversarial Network for molecules
            - Generates realistic molecular structures
            - Good for diversity and quality
            """)
        
        with col3:
            st.markdown("""
            **Autoregressive**
            - Transformer-based sequence generation
            - Generates SMILES strings directly
            - Good for conditional generation
            """)
        
        return
    
    # Model loaded - show generation interface
    st.subheader("Molecular Generation")
    
    # Generation controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("Generate Molecules", type="primary"):
            with st.spinner("Generating molecules..."):
                try:
                    # Set seed for reproducibility
                    set_seed(seed)
                    
                    # Generate samples
                    samples = st.session_state.sampler.sample_molecules(
                        num_samples=num_samples,
                        seed=seed,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                    st.session_state.generated_samples = samples
                    st.success(f"Generated {len(samples)} molecules!")
                    
                except Exception as e:
                    st.error(f"Generation failed: {e}")
    
    with col2:
        if st.button("Evaluate Metrics"):
            if 'generated_samples' in st.session_state:
                with st.spinner("Computing metrics..."):
                    try:
                        smiles_list = [sample['smiles'] for sample in st.session_state.generated_samples]
                        metrics = st.session_state.metrics.compute_all_metrics(smiles_list)
                        st.session_state.evaluation_metrics = metrics
                        st.success("Metrics computed!")
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
    
    with col3:
        if st.button("Save Samples"):
            if 'generated_samples' in st.session_state:
                try:
                    # Save as JSON
                    import json
                    samples_data = []
                    for sample in st.session_state.generated_samples:
                        samples_data.append({
                            'smiles': sample['smiles'],
                            'properties': sample['properties']
                        })
                    
                    json_str = json.dumps(samples_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="generated_molecules.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Save failed: {e}")
    
    # Display generated samples
    if 'generated_samples' in st.session_state:
        st.subheader("Generated Molecules")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Grid View", "List View", "Metrics"])
        
        with tab1:
            # Grid view of molecular images
            samples = st.session_state.generated_samples
            cols = 4
            rows = (len(samples) + cols - 1) // cols
            
            for row in range(rows):
                col_containers = st.columns(cols)
                for col in range(cols):
                    idx = row * cols + col
                    if idx < len(samples):
                        with col_containers[col]:
                            sample = samples[idx]
                            if sample['image'] is not None:
                                st.image(sample['image'], caption=f"Sample {idx+1}")
                                st.text(f"SMILES: {sample['smiles']}")
                            else:
                                st.text(f"Sample {idx+1}")
                                st.text(f"SMILES: {sample['smiles']}")
        
        with tab2:
            # List view with detailed information
            for i, sample in enumerate(samples):
                with st.expander(f"Sample {i+1}: {sample['smiles']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if sample['image'] is not None:
                            st.image(sample['image'], width=300)
                    
                    with col2:
                        st.subheader("Properties")
                        if sample['properties']:
                            for prop, value in sample['properties'].items():
                                st.metric(prop.replace('_', ' ').title(), f"{value:.3f}")
                        else:
                            st.text("No properties computed")
        
        with tab3:
            # Metrics view
            if 'evaluation_metrics' in st.session_state:
                metrics = st.session_state.evaluation_metrics
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Validity", f"{metrics.get('validity', 0):.3f}")
                with col2:
                    st.metric("Uniqueness", f"{metrics.get('uniqueness', 0):.3f}")
                with col3:
                    st.metric("Novelty", f"{metrics.get('novelty', 0):.3f}")
                with col4:
                    st.metric("Lipinski Compliance", f"{metrics.get('lipinski_compliance', 0):.3f}")
                
                # Detailed metrics
                st.subheader("Detailed Metrics")
                
                # QED scores
                if 'qed_scores' in metrics and metrics['qed_scores']:
                    st.subheader("QED Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(metrics['qed_scores'], bins=20, alpha=0.7, color='blue')
                    ax.set_xlabel('QED Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('QED Score Distribution')
                    st.pyplot(fig)
                
                # LogP scores
                if 'logp_scores' in metrics and metrics['logp_scores']:
                    st.subheader("LogP Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(metrics['logp_scores'], bins=20, alpha=0.7, color='green')
                    ax.set_xlabel('LogP')
                    ax.set_ylabel('Frequency')
                    ax.set_title('LogP Distribution')
                    st.pyplot(fig)
                
                # Molecular weight
                if 'mw_scores' in metrics and metrics['mw_scores']:
                    st.subheader("Molecular Weight Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(metrics['mw_scores'], bins=20, alpha=0.7, color='red')
                    ax.set_xlabel('Molecular Weight')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Molecular Weight Distribution')
                    st.pyplot(fig)
            else:
                st.info("Click 'Evaluate Metrics' to compute molecular properties")
    
    # Interpolation section (for VAE models)
    if isinstance(st.session_state.model, GraphVAE):
        st.subheader("Molecular Interpolation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Select two molecules for interpolation:")
            if 'generated_samples' in st.session_state:
                sample_options = [f"Sample {i+1}: {sample['smiles']}" for i, sample in enumerate(st.session_state.generated_samples)]
                
                mol1_idx = st.selectbox("First Molecule", range(len(sample_options)), format_func=lambda x: sample_options[x])
                mol2_idx = st.selectbox("Second Molecule", range(len(sample_options)), format_func=lambda x: sample_options[x])
                
                num_steps = st.slider("Interpolation Steps", 3, 20, 10)
                
                if st.button("Interpolate"):
                    if mol1_idx != mol2_idx:
                        with st.spinner("Interpolating..."):
                            try:
                                mol1 = st.session_state.generated_samples[mol1_idx]
                                mol2 = st.session_state.generated_samples[mol2_idx]
                                
                                interpolated = st.session_state.sampler.interpolate_molecules(
                                    mol1, mol2, num_steps, seed
                                )
                                
                                st.session_state.interpolated_samples = interpolated
                                st.success(f"Generated {len(interpolated)} interpolated molecules!")
                                
                            except Exception as e:
                                st.error(f"Interpolation failed: {e}")
                    else:
                        st.warning("Please select different molecules for interpolation")
        
        with col2:
            if 'interpolated_samples' in st.session_state:
                st.subheader("Interpolation Results")
                
                interpolated = st.session_state.interpolated_samples
                cols = 5
                rows = (len(interpolated) + cols - 1) // cols
                
                for row in range(rows):
                    col_containers = st.columns(cols)
                    for col in range(cols):
                        idx = row * cols + col
                        if idx < len(interpolated):
                            with col_containers[col]:
                                sample = interpolated[idx]
                                if sample['image'] is not None:
                                    st.image(sample['image'], width=100)
                                st.text(f"α={sample['alpha']:.2f}")

if __name__ == "__main__":
    main()
