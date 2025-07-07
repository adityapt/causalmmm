#!/usr/bin/env python3
"""
Causal MMM Streamlit Application
A web interface for training and analyzing Causal Marketing Mix Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import json
import os
import tempfile
from datetime import datetime
import warnings
import time
import threading
from queue import Queue
import sys
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def format_number_units(value):
    """Format large numbers with K/M units for better readability"""
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return f"{value:.2f}"

# Import the ML model components
sys.path.append('ml_backend')

# Set page config
st.set_page_config(
    page_title="Causal MMM Training",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e9ecef;
}
.stProgress .stProgress > div > div > div > div {
    background-color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

def calculate_correlation_matrix(df, selected_vars):
    """Calculate correlation matrix for selected variables"""
    if len(selected_vars) < 2:
        return None
    
    # Select only numeric columns
    numeric_df = df[selected_vars].select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return None
    
    return numeric_df.corr()

def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap using plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Variable Correlation Matrix",
        xaxis_title="Variables",
        yaxis_title="Variables",
        height=500
    )
    
    return fig

def create_distribution_plots(df, variables, var_type):
    """Create distribution plots for selected variables"""
    if not variables:
        return None
    
    # Create subplots
    n_vars = len(variables)
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=variables,
        vertical_spacing=0.1
    )
    
    for i, var in enumerate(variables):
        row = i // cols + 1
        col = i % cols + 1
        
        if var in df.columns and df[var].dtype in ['int64', 'float64']:
            fig.add_trace(
                go.Histogram(x=df[var], name=var, showlegend=False),
                row=row, col=col
            )
    
    fig.update_layout(
        title=f"{var_type} Variable Distributions",
        height=200 * rows,
        showlegend=False
    )
    
    return fig

def create_time_series_plot(df, date_col, variables):
    """Create time series plot for variables"""
    if not variables or date_col not in df.columns:
        return None
    
    # Convert date column to datetime
    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    
    fig = go.Figure()
    
    for var in variables:
        if var in df.columns and df[var].dtype in ['int64', 'float64']:
            fig.add_trace(go.Scatter(
                x=df_plot[date_col],
                y=df_plot[var],
                mode='lines',
                name=var,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Time Series of Selected Variables",
        xaxis_title="Date",
        yaxis_title="Values",
        height=400
    )
    
    return fig

def main():
    st.title("🎯 Causal Marketing Mix Modeling")
    st.markdown("Train advanced ML models for marketing attribution analysis with real-time outputs")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Initialize selected tab if not present
    if 'selected_tab' not in st.session_state:
        st.session_state['selected_tab'] = "Data Upload & Setup"
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Upload & Setup", "Data Analysis", "Model Training", "Results"],
        index=["Data Upload & Setup", "Data Analysis", "Model Training", "Results"].index(st.session_state['selected_tab'])
    )
    
    # Update session state when user changes tab
    if page != st.session_state['selected_tab']:
        st.session_state['selected_tab'] = page
    
    if page == "Data Upload & Setup":
        data_upload_page()
    elif page == "Data Analysis":
        data_analysis_page()
    elif page == "Model Training":
        training_page()
    elif page == "Results":
        results_page()

def data_upload_page():
    st.header("📊 Data Upload & Variable Selection")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file with marketing data",
        type="csv",
        help="Upload your marketing data with time series information"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to session state
        df = pd.read_csv(uploaded_file)
        st.session_state['raw_data'] = df
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Variable selection
        st.subheader("🎯 Variable Selection")
        st.markdown("Select variables for your MMM model:")
        
        # Get all columns
        all_columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Marketing variables (media spend)
            st.write("**📺 Marketing Variables** (Media Spend)")
            marketing_vars = st.multiselect(
                "Select marketing/media variables",
                options=all_columns,
                default=[col for col in all_columns if any(keyword in col.lower() for keyword in ['spend', 'media', 'tv', 'digital', 'radio', 'social'])],
                help="Variables representing marketing spend across different channels"
            )
            
            # Control variables
            st.write("**⚙️ Control Variables**")
            control_vars = st.multiselect(
                "Select control variables",
                options=all_columns,
                default=[col for col in all_columns if col not in marketing_vars and any(keyword in col.lower() for keyword in ['control', 'season', 'trend', 'macro', 'external'])],
                help="Variables representing external factors that influence the dependent variable"
            )
        
        with col2:
            # Dependent variable
            st.write("**🎯 Dependent Variable** (Target)")
            dependent_var = st.selectbox(
                "Select dependent variable",
                options=[col for col in all_columns if col not in marketing_vars + control_vars],
                index=0 if [col for col in all_columns if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'conversion'])] else 0,
                help="The variable you want to predict (e.g., revenue, sales)"
            )
            
            # Date and region variables
            st.write("**📅 Date Variable**")
            date_var = st.selectbox(
                "Select date variable",
                options=[None] + [col for col in all_columns if any(keyword in col.lower() for keyword in ['date', 'time', 'week', 'month'])],
                help="Time variable for time series analysis"
            )
            
            st.write("**🌍 Region Variable** (Optional)")
            region_var = st.selectbox(
                "Select region variable",
                options=[None] + [col for col in all_columns if any(keyword in col.lower() for keyword in ['region', 'geo', 'country', 'state'])],
                help="Regional variable for multi-region analysis"
            )
        
        # Store variable selections
        if marketing_vars and dependent_var:
            st.session_state['variable_config'] = {
                'marketing_vars': marketing_vars,
                'control_vars': control_vars,
                'dependent_var': dependent_var,
                'date_var': date_var,
                'region_var': region_var
            }
            
            # Show summary
            st.subheader("📋 Variable Summary")
            summary_data = {
                'Variable Type': ['Marketing', 'Control', 'Dependent', 'Date', 'Region'],
                'Count': [len(marketing_vars), len(control_vars), 1, 1 if date_var else 0, 1 if region_var else 0],
                'Variables': [
                    ', '.join(marketing_vars[:3]) + ('...' if len(marketing_vars) > 3 else ''),
                    ', '.join(control_vars[:3]) + ('...' if len(control_vars) > 3 else ''),
                    dependent_var,
                    date_var or 'Not selected',
                    region_var or 'Not selected'
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            st.success("✅ Variables configured! Ready to explore your data.")
            
            # Add navigation button
            if st.button("🔍 Go to Data Analysis", type="primary", use_container_width=True):
                st.session_state['selected_tab'] = 'Data Analysis'
                st.rerun()
        else:
            st.warning("Please select at least one marketing variable and one dependent variable.")
    
    else:
        st.info("Please upload a CSV file to get started.")

def data_analysis_page():
    st.header("📊 Data Analysis & Exploration")
    
    if 'raw_data' not in st.session_state or 'variable_config' not in st.session_state:
        st.warning("Please upload data and configure variables first.")
        return
    
    df = st.session_state['raw_data']
    config = st.session_state['variable_config']
    
    # Basic statistics
    st.subheader("📈 Basic Statistics")
    
    # Select variables to analyze
    all_selected_vars = (config['marketing_vars'] + 
                        config['control_vars'] + 
                        [config['dependent_var']])
    
    if config['date_var']:
        all_selected_vars.append(config['date_var'])
    if config['region_var']:
        all_selected_vars.append(config['region_var'])
    
    # Filter numeric variables for statistics
    numeric_vars = [var for var in all_selected_vars if var in df.select_dtypes(include=[np.number]).columns]
    
    if numeric_vars:
        stats_df = df[numeric_vars].describe()
        st.dataframe(stats_df, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    if len(numeric_vars) >= 2:
        corr_matrix = calculate_correlation_matrix(df, numeric_vars)
        if corr_matrix is not None:
            fig = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
            
            if strong_corr:
                st.warning("**Strong correlations detected:**")
                for corr in strong_corr:
                    st.write(f"• {corr}")
    
    # Variable distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📺 Marketing Variable Distributions")
        marketing_numeric = [var for var in config['marketing_vars'] if var in numeric_vars]
        if marketing_numeric:
            fig = create_distribution_plots(df, marketing_numeric, "Marketing")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Dependent Variable Distribution")
        if config['dependent_var'] in numeric_vars:
            fig = px.histogram(
                df, 
                x=config['dependent_var'],
                title=f"Distribution of {config['dependent_var']}",
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    if config['date_var']:
        st.subheader("⏰ Time Series Analysis")
        
        # Plot marketing variables over time
        fig = create_time_series_plot(df, config['date_var'], config['marketing_vars'][:5])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Plot dependent variable over time
        fig2 = create_time_series_plot(df, config['date_var'], [config['dependent_var']])
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    
    # Regional analysis
    if config['region_var']:
        st.subheader("🌍 Regional Analysis")
        
        # Revenue by region
        if config['dependent_var'] in numeric_vars:
            region_summary = df.groupby(config['region_var'])[config['dependent_var']].agg(['sum', 'mean', 'count']).reset_index()
            region_summary.columns = [config['region_var'], 'Total', 'Average', 'Count']
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(region_summary, x=config['region_var'], y='Total', 
                           title=f"Total {config['dependent_var']} by Region")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(region_summary, x=config['region_var'], y='Average',
                           title=f"Average {config['dependent_var']} by Region")
                st.plotly_chart(fig, use_container_width=True)
    
    # Marketing spend vs dependent variable scatter
    st.subheader("💰 Marketing Spend vs Outcome")
    
    if config['marketing_vars'] and config['dependent_var'] in numeric_vars:
        # Calculate total marketing spend
        marketing_numeric = [var for var in config['marketing_vars'] if var in numeric_vars]
        if marketing_numeric:
            df['total_marketing_spend'] = df[marketing_numeric].sum(axis=1)
            
            fig = px.scatter(
                df,
                x='total_marketing_spend',
                y=config['dependent_var'],
                title=f"Total Marketing Spend vs {config['dependent_var']}"
            )
            st.plotly_chart(fig, use_container_width=True)

def training_page():
    st.header("🚀 Model Training with Real-time Outputs")
    
    if 'raw_data' not in st.session_state or 'variable_config' not in st.session_state:
        st.warning("Please upload data and configure variables first.")
        return
    
    df = st.session_state['raw_data']
    config = st.session_state['variable_config']
    
    # Training parameters
    st.subheader("⚙️ Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input("Epochs", min_value=50, max_value=1000, value=300, step=50,
                                help="Number of training iterations")
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f",
                                       help="Model learning speed")
    
    with col2:
        hidden_size = st.number_input("Hidden Size", min_value=32, max_value=128, value=64, step=16,
                                     help="Neural network complexity")
        train_ratio = st.number_input("Train Ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.1,
                                     help="Training data percentage")
    
    with col3:
        seed = st.number_input("Random Seed", min_value=1, max_value=10000, value=42, step=1,
                              help="Reproducibility seed")
        smoothness_lambda = st.number_input("Temporal Smoothness (λ)", min_value=0.0001, max_value=0.05, value=0.01, step=0.0001, format="%.4f",
                                           help="Penalty for week-0 spikes (higher = smoother coefficients)")
        burn_in_weeks = st.slider("Synthetic burn-in weeks", min_value=0, max_value=10, value=4,
                                 help="Number of synthetic weeks to add before real data for GRU training stability")
    
    # Model configuration summary
    st.subheader("🏗️ Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Model Architecture:**
        • Causal GRU MMM with Bayesian Networks
        • Marketing Variables: {len(config['marketing_vars'])}
        • Control Variables: {len(config['control_vars'])}
        • Hidden Units: {hidden_size}
        • Regions: {len(df[config['region_var']].unique()) if config['region_var'] else 1}
        """)
    
    with col2:
        st.info(f"""
        **Data Configuration:**
        • Total Samples: {len(df)}
        • Training Samples: {int(len(df) * train_ratio)}
        • Test Samples: {len(df) - int(len(df) * train_ratio)}
        • Target Variable: {config['dependent_var']}
        """)
    
    # Training button and real-time display
    if st.button("🎯 Start Training", type="primary", use_container_width=True):
        
        # Create placeholders for real-time updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Metrics placeholders
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_loss = st.empty()
        with col2:
            metric_r2 = st.empty()
        with col3:
            metric_epoch = st.empty()
        with col4:
            metric_time = st.empty()
        
        # Plot placeholders
        plot_container = st.container()
        with plot_container:
            col1, col2 = st.columns(2)
            with col1:
                loss_plot = st.empty()
            with col2:
                coeff_plot = st.empty()
        
        try:
            # Prepare data for training
            status_text.text("🔄 Preparing data...")
            
            # Create the exact data structure expected by the model
            prepared_df = prepare_data_for_model(df, config)
            
            # Import and run the model with real-time updates
            from ml_backend.causal_gru_mmm import (
                gs, CausalEncoder, GRUCausalMMM, 
                create_belief_vectors, create_media_adjacency,
                prepare_data_for_training
            )
            
            # Set seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
            
            # Use improved data preparation function
            status_text.text("🧠 Preparing data with proper scaling...")
            
            # Create belief vectors and adjacency matrix
            Z_ctrl, bn_struct = create_belief_vectors(prepared_df, config.get('control_vars', []))
            A_media = create_media_adjacency(config.get('marketing_vars', []), bn_struct)
            
            # Add burn-in parameter to config
            config_with_burnin = config.copy()
            config_with_burnin["burn_in_weeks"] = burn_in_weeks
            
            # Use the improved data preparation function
            data_result = prepare_data_for_training(prepared_df, config_with_burnin)
            X_m = data_result['X_m']
            X_c = data_result['X_c'] 
            R = data_result['R']
            y = data_result['y']
            # Extract scalers for proper inverse transformations
            media_scaler = data_result['media_scaler']
            control_scaler = data_result['control_scaler']
            y_scaler = data_result['y_scaler']
            burn_in = data_result['burn_in']  # Number of synthetic weeks added
            
            # Train/test split - Use stratified sampling to maintain distribution
            n_regions, n_time_steps = X_m.shape[0], X_m.shape[1]
            
            # For single region with trending data, use every 5th time step for test
            # This maintains the same distribution across train/test sets
            test_indices = np.arange(4, n_time_steps, 5)  # Every 5th starting from index 4
            train_indices = np.array([i for i in range(n_time_steps) if i not in test_indices])
            
            # Ensure we have roughly the desired train_ratio
            target_test_size = int(n_time_steps * (1 - train_ratio))
            if len(test_indices) > target_test_size:
                test_indices = test_indices[:target_test_size]
            
            # Update train indices after potentially reducing test indices
            train_indices = np.array([i for i in range(n_time_steps) if i not in test_indices])
            
            # Split data using indices
            Xm_tr, Xm_te = X_m[:, train_indices], X_m[:, test_indices]
            Xc_tr, Xc_te = X_c[:, train_indices], X_c[:, test_indices]
            Y_tr, Y_te = y[:, train_indices], y[:, test_indices]
            
            # The data is already properly scaled - media variables will be scaled after adstock
            # y_scaler is provided by prepare_data_for_training for inverse transformation
            
            # Convert to tensors directly (no additional scaling needed)
            Xm_tr = torch.tensor(Xm_tr.numpy(), dtype=torch.float32)
            Xc_tr = torch.tensor(Xc_tr.numpy(), dtype=torch.float32)
            Y_tr = torch.tensor(Y_tr.numpy(), dtype=torch.float32)
            
            Xm_te = torch.tensor(Xm_te.numpy(), dtype=torch.float32)
            Xc_te = torch.tensor(Xc_te.numpy(), dtype=torch.float32)
            Y_te = torch.tensor(Y_te.numpy(), dtype=torch.float32)
            
            R_ids = R
            
            # Initialize model with dynamic dimensions
            n_regions = X_m.shape[0]
            n_media = len(config['marketing_vars'])
            n_control = len(config['control_vars']) if config['control_vars'] else 1
            model = GRUCausalMMM(A_media, n_media=n_media, ctrl_dim=n_control, hidden=hidden_size, n_regions=n_regions)
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
            lossf = nn.MSELoss()
            
            # Training with real-time updates
            train_losses = []
            val_losses = []
            r2_scores = []
            coeff_history = []  # Track coefficient evolution
            start_time = time.time()
            
            status_text.text("🏋️ Training model...")
            
            for ep in range(epochs):
                model.train()
                y_hat, w_coeffs, media_contribs = model(Xm_tr, Xc_tr, R_ids)
                
                # Calculate MSE loss
                mse_loss = lossf(y_hat, Y_tr)
                
                # Add temporal smoothness penalty to fix week-0 spike and negative R²
                # This penalty encourages coefficients to be smooth over time
                temporal_penalty = 0.0
                if w_coeffs is not None and w_coeffs.shape[1] > 1:  # Only if we have multiple time steps
                    # Calculate difference between consecutive time steps: (w_t+1 - w_t)²
                    temporal_diff = w_coeffs[:, 1:, :] - w_coeffs[:, :-1, :]  # [regions, time-1, media]
                    temporal_penalty = temporal_diff.pow(2).mean()  # Mean squared temporal difference
                
                # Combined loss with smoothness regularization (user-defined λ)
                loss = mse_loss + smoothness_lambda * temporal_penalty
                
                opt.zero_grad()
                loss.backward()
                
                # Optional gradient clipping for week-0 stability (as recommended)
                if w_coeffs is not None and w_coeffs.shape[1] > 0:
                    # Clip gradients for the first timestep to prevent week-0 spikes
                    for name, param in model.named_parameters():
                        if param.grad is not None and 'gru' in name.lower():
                            param.grad.data.clamp_(-2, 2)
                
                opt.step()
                
                train_losses.append(loss.item())
                
                # Validation and real-time updates every 10 epochs
                if ep % 10 == 0 or ep == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        y_val, w_coeffs, media_contribs_val = model(Xm_te, Xc_te, R_ids)
                        val_loss = lossf(y_val, Y_te).item()
                        val_losses.append(val_loss)
                        
                        # Track coefficient evolution
                        if w_coeffs is not None:
                            coeff_history.append(w_coeffs.detach().cpu().numpy())
                        
                        # Calculate R2 on test data (both in scaled space for consistency)
                        y_pred_flat = y_val.cpu().numpy().flatten()
                        y_true_flat = Y_te.cpu().numpy().flatten()
                        
                        # Check for NaN values and handle them
                        if np.isnan(y_pred_flat).any() or np.isnan(y_true_flat).any():
                            r2 = -1.0  # Set a default value when NaN is present
                            st.warning(f"NaN values detected in epoch {ep + 1}, setting R² to -1.0")
                        else:
                            r2 = r2_score(y_true_flat, y_pred_flat)
                        r2_scores.append(r2)
                    
                    # Update metrics
                    progress = (ep + 1) / epochs
                    progress_bar.progress(progress)
                    
                    elapsed_time = time.time() - start_time
                    metric_loss.metric("Current Loss", f"{loss.item():.4f}")
                    metric_r2.metric("R² Score", f"{r2:.4f}")
                    metric_epoch.metric("Epoch", f"{ep + 1}/{epochs}")
                    metric_time.metric("Time", f"{elapsed_time:.1f}s")
                    
                    # Update loss plot
                    if len(train_losses) > 1:
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=train_losses,
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='blue')
                        ))
                        
                        if len(val_losses) > 1:
                            val_epochs = list(range(0, len(train_losses), 10))[:len(val_losses)]
                            fig_loss.add_trace(go.Scatter(
                                x=val_epochs,
                                y=val_losses,
                                mode='lines+markers',
                                name='Validation Loss',
                                line=dict(color='red')
                            ))
                        
                        fig_loss.update_layout(
                            title="Training Progress",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=300
                        )
                        loss_plot.plotly_chart(fig_loss, use_container_width=True)
                    
                    # Update coefficient plot (strip burn-in for display)
                    if w_coeffs is not None:
                        fig_coeff = go.Figure()
                        # Strip burn-in weeks from coefficient display
                        coeffs_to_display = w_coeffs[0, burn_in_weeks:, 0].detach().cpu().numpy() if burn_in_weeks > 0 else w_coeffs[0, :, 0].detach().cpu().numpy()
                        fig_coeff.add_trace(go.Scatter(
                            y=coeffs_to_display,
                            mode='lines',
                            name='β_media_1',
                            line=dict(color='green')
                        ))
                        fig_coeff.update_layout(
                            title="Media Coefficient Evolution (Region 0, Media 1)",
                            xaxis_title="Time Step",
                            yaxis_title="Coefficient Value",
                            height=300
                        )
                        coeff_plot.plotly_chart(fig_coeff, use_container_width=True)
            
            # Final inference on test set for metrics - BOTH UNSCALED FOR PROPER COMPARISON
            model.eval()
            with torch.no_grad():
                y_scaled_te, w_te, contrib_te = model(Xm_te, Xc_te, R_ids)
                # Inverse scale BOTH predicted and actual values to original scale
                y_pred = y_scaler.inverse_transform(y_scaled_te.cpu().numpy().reshape(-1,1)).reshape(n_regions, -1)
                y_test_actual = y_scaler.inverse_transform(Y_te.cpu().numpy().reshape(-1,1)).reshape(n_regions, -1)
                
                # Get coefficients and contributions for full timeline (training data has more time periods)
                y_full_pred, w_full, contrib_full = model(Xm_tr, Xc_tr, R_ids)
                full_timeline_coeffs = w_full.detach().cpu().numpy() if w_full is not None else w_te.detach().cpu().numpy()
                
                contrib_raw = contrib_full.detach().cpu().numpy()  # already in scaled space

                # Inverse-transform *all* contributions with the same y_scaler
                # used for the target, restoring business-unit $/kPI values.
                contrib_flat = contrib_raw.reshape(-1, 1)          # [B*T*n_media, 1]
                contrib_real  = y_scaler.inverse_transform(contrib_flat)          # back to original scale
                full_timeline_contribs = contrib_real.reshape(contrib_raw.shape)  # [B,T,n_media]
                
                # Drop the synthetic burn-in prefix so plots start at real Week-0
                if burn_in > 0:
                    y_pred = y_pred[:, burn_in:]
                    y_test_actual = y_test_actual[:, burn_in:]
                    full_timeline_coeffs = full_timeline_coeffs[:, burn_in:, :]
                    full_timeline_contribs = full_timeline_contribs[:, burn_in:, :]
            
            # Calculate final metrics
            mse = mean_squared_error(y_test_actual.flatten(), y_pred.flatten())
            rmse = np.sqrt(mse)
            final_r2 = r2_score(y_test_actual.flatten(), y_pred.flatten())
            mape = np.mean(np.abs((y_test_actual.flatten() - y_pred.flatten()) / (y_test_actual.flatten() + 1e-8))) * 100
            
            # Store results
            results = {
                'model_performance': {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2_score': float(final_r2),
                    'mape': float(mape)
                },
                'training_history': {
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                    'r2_scores': r2_scores
                },
                'model_coefficients': {
                    'adstock_params': model.alpha.detach().cpu().numpy().tolist(),
                    'saturation_params': {
                        'hill_a': model.hill_a.detach().cpu().numpy().tolist(),
                        'hill_g': model.hill_g.detach().cpu().numpy().tolist()
                    },
                    'media_weights': full_timeline_coeffs.tolist(),
                    'media_contributions': full_timeline_contribs.tolist()
                },
                'predictions': {
                    'y_pred': y_pred.tolist(),
                    'y_actual': y_test_actual.tolist()
                },
                'data_info': {
                    'n_regions': n_regions,
                    'media_vars': config.get('marketing_vars', []),
                    'control_vars': config.get('control_vars', []),
                    'train_samples': len(Y_tr.flatten()),
                    'test_samples': len(Y_te.flatten())
                }
            }
            
            st.session_state['training_results'] = results
            
            # Store model and additional data for combined analysis
            st.session_state['trained_model'] = model
            st.session_state['training_data'] = {
                'X_m': X_m,
                'X_c': X_c,
                'y': y,
                'R': R_ids,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'original_df': prepared_df,  # Add original dataframe for proper scaling
                'target_col': config['dependent_var'],  # Add target column name
                'y_scaler': y_scaler,  # Add y scaler for inverse transformation
                'burn_in': burn_in,  # Add burn-in value for trimming
                'X_m_scaled': Xm_tr,  # Store scaled training data
                'X_c_scaled': Xc_tr,  # Store scaled training data
                'Xm_te_scaled': Xm_te,  # Store scaled test data  
                'Xc_te_scaled': Xc_te   # Store scaled test data
            }
            
            status_text.text("✅ Training completed successfully!")
            st.success(f"🎉 Model trained! Final R² Score: {final_r2:.4f}")
            st.balloons()
            
            # Store coefficient history in session state
            st.session_state['coeff_history'] = coeff_history
            st.session_state['n_regions'] = n_regions
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.write("Error details:", e)
    
    # Coefficient visualization section (persistent after training)
    if 'coeff_history' in st.session_state and 'n_regions' in st.session_state:
        st.subheader("📊 Media Coefficient Evolution")
        
        coeff_history = st.session_state['coeff_history']
        n_regions = st.session_state['n_regions']
        
        # Create region and media selection dropdowns
        col1, col2 = st.columns(2)
        
        with col1:
            region_options = [f"Region {i}" for i in range(n_regions)]
            selected_region = st.selectbox(
                "Select Region:",
                region_options,
                key="training_region_select"
            )
            region_idx = int(selected_region.split()[1])
        
        with col2:
            # Use actual marketing variable names from config
            media_options = config['marketing_vars']
            selected_media = st.selectbox(
                "Select Media Variable:",
                media_options,
                key="training_media_select"
            )
            media_idx = media_options.index(selected_media)
        
        # Plot coefficient evolution for selected region and media
        if len(coeff_history) > 0:
            fig = go.Figure()
            
            # Extract coefficients for selected region and media
            coeff_evolution = []
            for epoch_coeffs in coeff_history:
                if epoch_coeffs is not None and len(epoch_coeffs) > region_idx:
                    region_coeffs = epoch_coeffs[region_idx]
                    if len(region_coeffs) > media_idx:
                        coeff_evolution.append(region_coeffs[media_idx])
                    else:
                        coeff_evolution.append(0)
                else:
                    coeff_evolution.append(0)
            
            fig.add_trace(go.Scatter(
                y=coeff_evolution,
                mode='lines+markers',
                name=f'{selected_region} - {selected_media}',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=f"Coefficient Evolution: {selected_region} - {selected_media}",
                xaxis_title="Epoch",
                yaxis_title="Coefficient Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def prepare_data_for_model(df, config):
    """Prepare data in the format expected by the ML model"""
    prepared_df = df.copy()
    
    # Map marketing variables to media_1, media_2, etc.
    for i, var in enumerate(config['marketing_vars']):
        if i < 10:  # Model expects exactly 10 media variables
            prepared_df[f'media_{i+1}'] = prepared_df[var]
    
    # Fill remaining media variables if needed
    for i in range(len(config['marketing_vars']), 10):
        prepared_df[f'media_{i+1}'] = np.random.rand(len(prepared_df)) * 100
    
    # Map control variables
    for i, var in enumerate(config['control_vars']):
        if i < 15:  # Model expects exactly 15 control variables
            prepared_df[f'control_{i+1}'] = prepared_df[var]
    
    # Fill remaining control variables if needed
    for i in range(len(config['control_vars']), 15):
        prepared_df[f'control_{i+1}'] = np.random.randint(0, 5, len(prepared_df))
    
    # Set target variable as 'sales'
    prepared_df['sales'] = prepared_df[config['dependent_var']]
    
    # Handle region
    if config['region_var']:
        prepared_df['region'] = prepared_df[config['region_var']]
    else:
        prepared_df['region'] = 'North'  # Default region
    
    # Handle date/week
    if config['date_var']:
        prepared_df['week'] = pd.to_datetime(prepared_df[config['date_var']]).dt.isocalendar().week
    else:
        prepared_df['week'] = np.arange(len(prepared_df))
    
    return prepared_df

def results_page():
    st.header("📈 Training Results & Analysis")
    
    if 'training_results' not in st.session_state:
        st.warning("No training results available. Please train a model first.")
        return
    
    results = st.session_state['training_results']
    
    # Performance metrics
    st.subheader("🎯 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    perf = results['model_performance']
    with col1:
        st.metric("R² Score", f"{perf['r2_score']:.4f}", help="Coefficient of determination")
    with col2:
        st.metric("RMSE", format_number_units(perf['rmse']), help="Root Mean Square Error")
    with col3:
        st.metric("MAPE", f"{perf['mape']:.2f}%", help="Mean Absolute Percentage Error")
    with col4:
        st.metric("MSE", format_number_units(perf['mse']), help="Mean Square Error")
    
    # Training visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Training History")
        history = results['training_history']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history['train_loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='blue')
        ))
        
        if 'val_loss' in history and history['val_loss']:
            val_epochs = list(range(0, len(history['train_loss']), 10))[:len(history['val_loss'])]
            fig.add_trace(go.Scatter(
                x=val_epochs,
                y=history['val_loss'],
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title="Loss During Training",
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Predictions vs Actual")
        pred_data = results['predictions']
        y_pred = np.array(pred_data['y_pred']).flatten()
        y_actual = np.array(pred_data['y_actual']).flatten()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_actual,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=y_actual,
                colorscale='viridis',
                size=8,
                opacity=0.7
            )
        ))
        
        # Add perfect prediction line
        min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Predicted vs Actual Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model parameters
    st.subheader("🔧 Learned Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Adstock Parameters (Decay Rates):**")
        coeffs = results['model_coefficients']
        adstock_df = pd.DataFrame({
            'Media Channel': [f'media_{i+1}' for i in range(len(coeffs['adstock_params']))],
            'Adstock Decay': coeffs['adstock_params']
        })
        st.dataframe(adstock_df, use_container_width=True)
        
        # Adstock visualization
        fig = px.bar(
            adstock_df,
            x='Media Channel',
            y='Adstock Decay',
            title="Adstock Decay Parameters by Channel"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Saturation Parameters:**")
        sat_params = coeffs['saturation_params']
        sat_df = pd.DataFrame({
            'Media Channel': [f'media_{i+1}' for i in range(len(sat_params['hill_a']))],
            'Hill Alpha': sat_params['hill_a'],
            'Hill Gamma': sat_params['hill_g']
        })
        st.dataframe(sat_df, use_container_width=True)
        
        # Saturation visualization
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=sat_df['Media Channel'],
            y=sat_df['Hill Alpha'],
            name='Hill Alpha',
            marker_color='lightblue'
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=sat_df['Media Channel'],
            y=sat_df['Hill Gamma'],
            mode='lines+markers',
            name='Hill Gamma',
            line=dict(color='red')
        ), secondary_y=True)
        
        fig.update_layout(title="Saturation Parameters")
        fig.update_yaxes(title_text="Hill Alpha", secondary_y=False)
        fig.update_yaxes(title_text="Hill Gamma", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Media coefficient evolution
    st.subheader("📈 Media Coefficient Evolution")
    if 'media_weights' in coeffs:
        media_weights = np.array(coeffs['media_weights'])
        
        # Add region and media selection dropdowns
        col1, col2 = st.columns(2)
        
        with col1:
            n_regions = media_weights.shape[0]
            region_options = [f"Region {i}" for i in range(n_regions)]
            selected_region = st.selectbox(
                "Select Region:",
                region_options,
                key="results_region_select"
            )
            region_idx = int(selected_region.split()[1])
        
        with col2:
            n_media = media_weights.shape[2]
            media_display_options = ["All Media Variables", "Select Specific Variables"]
            display_mode = st.selectbox(
                "Display Mode:",
                media_display_options,
                key="results_media_display_mode"
            )
        
        # Media selection based on display mode
        if display_mode == "Select Specific Variables":
            media_options = results['data_info']['media_vars']
            selected_media = st.multiselect(
                "Select Media Variables:",
                media_options,
                default=media_options[:min(5, len(media_options))],  # Default to first 5 or all if less than 5
                key="results_media_multiselect"
            )
            media_indices = [media_options.index(media) for media in selected_media]
        else:
            # Show all media variables
            media_indices = list(range(n_media))
            selected_media = results['data_info']['media_vars']
        
        # Plot coefficient evolution for selected region and media variables
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Always show the full 105-week timeline (burn-in already trimmed in coefficients)
        for idx, media_idx in enumerate(media_indices):
            color = colors[idx % len(colors)]
            
            # Get the full coefficient timeline (already trimmed of burn-in weeks)
            # media_weights is already [regions, trimmed_time, media] where trimmed_time = 105 weeks
            if n_regions == 1:
                # Single region: show full 105 weeks
                coeff_timeline = media_weights[0, :, media_idx]
            else:
                # Multiple regions: concatenate to show full timeline
                full_timeline = []
                for region in range(n_regions):
                    region_coeffs = media_weights[region, :, media_idx]
                    full_timeline.extend(region_coeffs)
                coeff_timeline = full_timeline
            
            # Use actual channel name instead of generic "Media {i+1}"
            media_name = results['data_info']['media_vars'][media_idx] if media_idx < len(results['data_info']['media_vars']) else f'Media {media_idx+1}'
            fig.add_trace(go.Scatter(
                y=coeff_timeline,
                mode='lines',
                name=media_name,
                line=dict(width=2, color=color)
            ))
        
        # Calculate the actual timeline length shown
        if n_regions == 1:
            timeline_length = media_weights.shape[1]  # Should be 105 weeks
        else:
            timeline_length = n_regions * media_weights.shape[1]
        
        title = f"Media Coefficient Evolution - Full Timeline ({timeline_length} weeks) - {len(media_indices)} Variables"
        xaxis_title = "Week"
        
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Coefficient Value",
            height=400,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Combined Train/Test Performance and Marketing Contributions
    st.subheader("🎯 Combined Train/Test Performance & Marketing Contributions")
    
    # Use the exact same data and results from the Results tab
    if 'training_results' in st.session_state and st.session_state['training_results']:
        try:
            results = st.session_state['training_results']
            config = st.session_state['variable_config']
            
            # Use the same predictions that are already calculated and shown in Results tab
            # These are already inverse-transformed to original scale
            pred_data = results['predictions']
            y_pred_test = np.array(pred_data['y_pred']).flatten()  # Test predictions (already in original scale)
            y_actual_test = np.array(pred_data['y_actual']).flatten()  # Test actual values (already in original scale)
            
            # Get the model and training data for additional calculations
            if ('trained_model' in st.session_state and 'training_data' in st.session_state):
                model = st.session_state['trained_model']
                training_data = st.session_state['training_data']
                
                X_m = training_data['X_m']
                X_c = training_data['X_c']
                y = training_data['y']
                R = training_data['R']
                train_indices = training_data['train_indices']
                test_indices = training_data['test_indices']
                
                # Get full timeline data for plotting - show all 105 weeks
                original_df = training_data.get('original_df')
                y_scaler = training_data.get('y_scaler')
                burn_in = training_data.get('burn_in', 0)
                
                if original_df is not None and y_scaler is not None:
                    # Get the target column values for ALL indices (original scale) excluding burn-in
                    target_col = training_data.get('target_col', 'conversions')
                    
                    # Create full timeline (105 weeks) combining train and test
                    full_indices = list(range(len(original_df)))
                    if burn_in > 0:
                        full_indices = full_indices[burn_in:]  # Remove burn-in weeks
                    
                    y_full_original = original_df[target_col].iloc[full_indices].values
                    
                    # Get model predictions for full timeline using model inference
                    model.eval()
                    with torch.no_grad():
                        # Run model on full input (already includes burn-in)
                        y_pred_full_scaled, _, _ = model(X_m, X_c, R)
                        
                        # Trim burn-in weeks and inverse transform to original scale
                        y_pred_full_trimmed = y_pred_full_scaled[:, burn_in:] if burn_in > 0 else y_pred_full_scaled
                        y_pred_full_np = y_pred_full_trimmed.detach().cpu().numpy().reshape(-1, 1)
                        y_pred_full_original = y_scaler.inverse_transform(y_pred_full_np).flatten()
                    
                    # Split into train and test for coloring but show full timeline
                    train_size = len(train_indices)
                    y_train_np = y_full_original[:train_size]
                    y_pred_train_np = y_pred_full_original[:train_size]
                    y_test_np = y_full_original[train_size:]
                    y_pred_test_np = y_pred_full_original[train_size:]
                    
                    # Create full timeline (0 to 104 for 105 weeks)
                    train_timeline = list(range(train_size))
                    test_timeline = list(range(train_size, len(y_full_original)))
                else:
                    # Fallback: use only available test data
                    y_train_np = np.array([])
                    y_pred_train_np = np.array([])
                    train_timeline = []
                    test_timeline = list(range(len(y_pred_test)))
                
                # Get test data for plotting - inverse scale both predictions and actual values
                # Both training and test data should be in the original scale for proper comparison
                Xm_te_scaled = training_data.get('Xm_te_scaled')
                Xc_te_scaled = training_data.get('Xc_te_scaled')
                
                if original_df is not None and y_scaler is not None:
                    # Get the target column values for test indices (original scale)
                    target_col = training_data.get('target_col', 'conversions')
                    y_test_original = original_df[target_col].iloc[test_indices].values
                    
                    # Get model predictions for test data using SCALED inputs
                    # (since model was trained on scaled data)
                    model.eval()
                    with torch.no_grad():
                        # Get test predictions on scaled data (model's native scale)
                        y_pred_test_scaled, _, _ = model(Xm_te_scaled, Xc_te_scaled, R)
                        
                        # Inverse transform predictions back to original scale
                        y_pred_test_scaled_np = y_pred_test_scaled.detach().cpu().numpy().reshape(-1, 1)
                        y_pred_test_original = y_scaler.inverse_transform(y_pred_test_scaled_np).flatten()
                    
                    # Both are now in original scale
                    y_test_np = y_test_original
                    y_pred_test_np = y_pred_test_original
                else:
                    # Fallback: use test data from Results tab
                    y_test_np = y_actual_test
                    y_pred_test_np = y_pred_test
            else:
                # Fallback: use only test data from Results tab
                train_timeline = []
                test_timeline = list(range(len(y_pred_test)))
                y_train_np = np.array([])
                y_pred_train_np = np.array([])
                y_test_np = y_actual_test
                y_pred_test_np = y_pred_test
            
            # Plot 1: Combined Train/Test Performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Combined Train/Test Performance Over Time**")
                fig = go.Figure()
                
                # Add training data
                fig.add_trace(go.Scatter(
                    x=train_timeline,
                    y=y_train_np,
                    mode='lines+markers',
                    name='Actual (Train)',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                fig.add_trace(go.Scatter(
                    x=train_timeline,
                    y=y_pred_train_np,
                    mode='lines+markers',
                    name='Predicted (Train)',
                    line=dict(color='lightblue', width=2),
                    marker=dict(size=4)
                ))
                
                # Add test data
                fig.add_trace(go.Scatter(
                    x=test_timeline,
                    y=y_test_np,
                    mode='lines+markers',
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=test_timeline,
                    y=y_pred_test_np,
                    mode='lines+markers',
                    name='Predicted (Test)',
                    line=dict(color='lightcoral', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Model Performance: Train vs Test Over Time",
                    xaxis_title="Time Step",
                    yaxis_title="Target Value",
                    height=400,
                    legend=dict(orientation="v")
                )
                st.plotly_chart(fig, use_container_width=True)
            

            
            # Add performance comparison in a new section
            st.write("**📊 Performance Comparison**")
            
            # Calculate metrics using the properly scaled data for both train and test
            # This ensures consistent scaling between train and test comparisons
            
            # Calculate metrics for training data (using original scale)
            train_r2 = r2_score(y_train_np, y_pred_train_np) if len(y_train_np) > 0 else 0.0
            train_mse = mean_squared_error(y_train_np, y_pred_train_np) if len(y_train_np) > 0 else 0.0
            
            # Calculate metrics for test data (using original scale)
            test_r2 = r2_score(y_test_np, y_pred_test_np) if len(y_test_np) > 0 else 0.0
            test_mse = mean_squared_error(y_test_np, y_pred_test_np) if len(y_test_np) > 0 else 0.0
            
            # Create comparison dataframe
            comparison_data = {
                'Metric': ['R² Score', 'MSE', 'RMSE', 'Data Points'],
                'Training': [
                    f"{train_r2:.4f}",
                    f"{train_mse:.2f}",
                    f"{np.sqrt(train_mse):.2f}",
                    f"{len(y_train_np)}"
                ],
                'Test': [
                    f"{test_r2:.4f}",
                    f"{test_mse:.2f}",
                    f"{np.sqrt(test_mse):.2f}",
                    f"{len(y_test_np)}"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Add interpretation
            st.write("**📝 Interpretation:**")
            if test_r2 > 0.7:
                st.success("✅ Excellent model performance")
            elif test_r2 > 0.5:
                st.info("✅ Good model performance")
            elif test_r2 > 0.3:
                st.warning("⚠️ Moderate model performance")
            else:
                st.error("❌ Poor model performance")
            
            # Show train/test gap
            performance_gap = abs(train_r2 - test_r2)
            if performance_gap < 0.1:
                st.success("✅ Low overfitting risk")
            elif performance_gap < 0.2:
                st.info("ℹ️ Moderate overfitting")
            else:
                st.warning("⚠️ High overfitting detected")
            
            # Plot 2: Marketing Channel Contributions Over Time
            st.write("**📈 Marketing Channel Contributions Over Time**")
            
            # Use the actual media contributions instead of coefficients
            coeffs = results['model_coefficients']
            training_data = st.session_state.get('training_data', {})
            burn_in = training_data.get('burn_in', 0)
            
            if 'media_contributions' in coeffs and coeffs['media_contributions']:
                # Use the actual media contributions (coefficient × transformed media value)
                media_contributions = np.array(coeffs['media_contributions'])
                # Strip burn-in weeks if present
                if burn_in > 0 and len(media_contributions.shape) >= 2:
                    if len(media_contributions.shape) == 3:  # [regions, time, media]
                        media_contributions = media_contributions[:, burn_in:, :]
                    elif len(media_contributions.shape) == 2:  # [time, media] 
                        media_contributions = media_contributions[burn_in:, :]
            elif 'media_weights' in coeffs and coeffs['media_weights']:
                # Fallback to coefficients if contributions not available
                media_contributions = np.array(coeffs['media_weights'])
                # Strip burn-in weeks if present
                if burn_in > 0 and len(media_contributions.shape) >= 2:
                    if len(media_contributions.shape) == 3:  # [regions, time, media]
                        media_contributions = media_contributions[:, burn_in:, :]
                    elif len(media_contributions.shape) == 2:  # [time, media]
                        media_contributions = media_contributions[burn_in:, :]
            else:
                media_contributions = None
            
            if media_contributions is not None:
                # Get original media variable names
                media_vars = config.get('marketing_vars', [])
                n_original_media = len(media_vars)
                
                # Safe channel count calculation BEFORE looping
                if len(media_contributions.shape) == 3:          # [R,T,C]
                    n_contrib_channels = media_contributions.shape[2]
                elif len(media_contributions.shape) == 2:        # [T,C]
                    n_contrib_channels = media_contributions.shape[1]
                else:
                    n_contrib_channels = 0
                
                max_channels = min(n_original_media, n_contrib_channels)
                
                # Additional safety check to prevent out-of-bounds
                if max_channels <= 0:
                    st.warning("No valid media channels found for contribution plotting")
                    max_channels = 0
                
                # Create contribution plot using actual media contributions
                fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                # Show only the original media variables  
                for i in range(max_channels):
                    # Extract contribution timeline for this media variable
                    if len(media_contributions.shape) == 3:  # [regions, time, media]
                        if media_contributions.shape[0] > 1:
                            contrib_timeline = media_contributions[:, :, i].sum(axis=0)  # Sum across regions
                        else:
                            contrib_timeline = media_contributions[0, :, i]
                    elif len(media_contributions.shape) == 2:  # [time, media]
                        contrib_timeline = media_contributions[:, i]
                    else:
                        continue
                    
                    media_name = media_vars[i] if i < len(media_vars) else f'Media {i+1}'
                    
                    fig.add_trace(go.Scatter(
                        y=contrib_timeline,
                        mode='lines',
                        name=media_name,
                        line=dict(width=2, color=colors[i % len(colors)]),
                        stackgroup='one'  # Create stacked area chart
                    ))
                
                fig.update_layout(
                    title="Marketing Channel Actual Contributions Over Time (Coeff × Media Value)",
                    xaxis_title="Time Step",
                    yaxis_title="Contribution Value",
                    height=400,
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show average contributions instead of coefficients
                st.write("**💰 Average Channel Contributions**")
                avg_contributions = []
                
                for i in range(max_channels):
                    if len(media_contributions.shape) == 3:  # [regions, time, media]
                        if media_contributions.shape[0] > 1:
                            avg_contrib = media_contributions[:, :, i].mean()
                        else:
                            avg_contrib = media_contributions[0, :, i].mean()
                    elif len(media_contributions.shape) == 2:  # [time, media]
                        avg_contrib = media_contributions[:, i].mean()
                    else:
                        avg_contrib = 0
                    avg_contributions.append(avg_contrib)
                
                if avg_contributions and sum(abs(c) for c in avg_contributions) != 0:
                    contrib_df = pd.DataFrame({
                        'Marketing Channel': media_vars[:len(avg_contributions)],
                        'Average Contribution': avg_contributions,
                        'Contribution %': [f"{(abs(c)/sum(abs(c2) for c2 in avg_contributions)*100):.1f}%" for c in avg_contributions]
                    })
                    
                    # Sort by contribution value
                    contrib_df = contrib_df.sort_values('Average Contribution', ascending=False)
                    st.dataframe(contrib_df, use_container_width=True)
                    
                    # Show top performer
                    if len(contrib_df) > 0:
                        top_channel = contrib_df.iloc[0]['Marketing Channel']
                        top_contrib = contrib_df.iloc[0]['Average Contribution']
                        st.info(f"🏆 **Top Performing Channel**: {top_channel} (avg contribution: {top_contrib:.4f})")
                else:
                    st.warning("No contribution data available for channel analysis.")
            else:
                st.warning("No media contribution data available. Please ensure model training completed successfully.")
        except Exception as e:
            st.error(f"Error generating combined analysis: {str(e)}")
            st.write("Please ensure the model has been trained successfully.")
    else:
        st.info("📝 **Train a model first** to see the combined train/test performance and marketing channel contributions.")
        st.write("Go to the 'Train Model' tab to run the training process, then return here to see the detailed analysis.")
    
    # Data summary
    st.subheader("📊 Data Summary")
    data_info = results['data_info']
    
    summary_data = {
        'Metric': ['Regions', 'Media Variables', 'Control Variables', 'Training Samples', 'Test Samples'],
        'Value': [
            data_info['n_regions'],
            len(data_info['media_vars']),
            len(data_info['control_vars']),
            data_info['train_samples'],
            data_info['test_samples']
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

if __name__ == "__main__":
    main()