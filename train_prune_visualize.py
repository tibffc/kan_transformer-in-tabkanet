# train_prune_visualize.py
"""
Обучает KAN-Transformer на одном фолде (10 эпох)
Применяет прунинг 20%
Сохраняет визуализации ДО и ПОСЛЕ прунинга
Поддерживает bankmarketing и multi_forest
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score, f1_score

from tabkanet.models import TabKANet
from tabkanet.tools import seed_everything, get_dataset, get_data_loader
from tabkanet.dataset import TabularDataset
from config import get_data_path

# ============== DATASET CONFIGURATION ==============

DATASET_CONFIG = {
    'bankmarketing': {
        'target': 'Output',
        'continuous': ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous'],
        'categorical': ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome'],
        'cont_names': ['Age (years)', 'Balance (EUR)', 'Duration (sec)', 'Campaign (# calls)', 'Days since last contact', 'Previous contacts'],
        'output_dim': 2,
        'task': 'binary'
    },
    'multi_forest': {
        'target': 'class',
        'continuous': ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'],
        'categorical': ['wilderness_area1', 'wilderness_area2', 'wilderness_area3', 'wilderness_area4', 'soil_type_1', 'soil_type_2', 'soil_type_3', 'soil_type_4', 'soil_type_5', 'soil_type_6', 'soil_type_7', 'soil_type_8', 'soil_type_9', 'soil_type_10', 'soil_type_11', 'soil_type_12', 'soil_type_13', 'soil_type_14', 'soil_type_15', 'soil_type_16', 'soil_type_17', 'soil_type_18', 'soil_type_19', 'soil_type_20', 'soil_type_21', 'soil_type_22', 'soil_type_23', 'soil_type_24', 'soil_type_25', 'soil_type_26', 'soil_type_27', 'soil_type_28', 'soil_type_29', 'soil_type_30', 'soil_type_31', 'soil_type_32', 'soil_type_33', 'soil_type_34', 'soil_type_35', 'soil_type_36', 'soil_type_37', 'soil_type_38', 'soil_type_39', 'soil_type_40'],
        'cont_names': ['Elevation', 'Aspect', 'Slope', 'Dist to Hydrology', 'Vert Dist Hydrology', 'Dist to Roadways', 'Hillshade 9am', 'Hillshade Noon', 'Hillshade 3pm', 'Dist to Fire Points'],
        'output_dim': 7,
        'task': 'multiclass'
    }
}

# ============== PRUNING FUNCTION ==============

def prune_kan_transformer(model, pruning_ratio=0.2):
    """Prune KAN-Transformer FFN layers"""
    if hasattr(model.transformer_model, 'blocks'):
        for block in model.transformer_model.blocks:
            if hasattr(block, 'kan_ffn'):
                kan_model = block.kan_ffn
                for layer in kan_model.layers:
                    if hasattr(layer, 'spline_weight'):
                        with torch.no_grad():
                            importance = torch.abs(layer.spline_weight).mean(dim=(0, 2))
                            threshold = torch.quantile(importance, pruning_ratio)
                            mask = (importance > threshold).float()
                            layer.spline_weight.data *= mask[None, :, None]
                            layer.base_weight.data *= mask[None, :]
    return model

# ============== VISUALIZATION FUNCTIONS ==============

def compute_kan_output(kan_layer, x_tensor, output_neuron=0):
    device = x_tensor.device
    kan_layer = kan_layer.to(device)
    with torch.no_grad():
        base_output = kan_layer.base_activation(x_tensor)
        base_out = torch.mm(base_output, kan_layer.base_weight.T)
        splines = kan_layer.b_splines(x_tensor)
        scaled_weight = kan_layer.scaled_spline_weight
        spline_out = torch.einsum('bic,oic->bo', splines, scaled_weight)
        output = base_out + spline_out
    return output[:, output_neuron].cpu().numpy()

def visualize_encoder_kan(model, continuous_features, cont_names, suffix=''):
    """Визуализация KAN Encoder"""
    kan_model = model.encoders['continuous_feature_encoder'].kanclassifier
    kan_layer = kan_model.layers[0]
    device = next(model.parameters()).device
    x_vals = np.linspace(-2, 2, 200)
    
    n_features = len(continuous_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for feat_idx, (feat_name, full_name) in enumerate(zip(continuous_features, cont_names)):
        y_vals = []
        for x in x_vals:
            x_tensor = torch.zeros(1, kan_layer.in_features, device=device)
            x_tensor[0, feat_idx] = x
            y = compute_kan_output(kan_layer, x_tensor, output_neuron=0)
            y_vals.append(y[0])
        
        axes[feat_idx].plot(x_vals, y_vals, 'b-', linewidth=2.5)
        axes[feat_idx].fill_between(x_vals, 0, y_vals, alpha=0.3, color='steelblue')
        axes[feat_idx].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[feat_idx].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        axes[feat_idx].set_title(full_name, fontsize=11)
        axes[feat_idx].set_xlabel('Normalized feature value', fontsize=10)
        axes[feat_idx].set_ylabel('Contribution to embedding', fontsize=10)
        axes[feat_idx].grid(True, alpha=0.3)
    
    for idx in range(feat_idx+1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'KAN Encoder: Continuous Feature Transformations {suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'kan_encoder{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: kan_encoder{suffix}.png")

def visualize_kan_transformer_ffn(model, suffix=''):
    """Визуализация KAN-Transformer FFN"""
    if not hasattr(model.transformer_model, 'blocks'):
        print("No KAN-Transformer blocks found!")
        return
    
    blocks = model.transformer_model.blocks
    device = next(model.parameters()).device
    
    for layer_idx, block in enumerate(blocks):
        if not hasattr(block, 'kan_ffn'):
            continue
        
        kan_ffn = block.kan_ffn
        kan_layer = kan_ffn.layers[0]
        
        spline_weights = kan_layer.spline_weight.detach().cpu().numpy()
        importance = np.abs(spline_weights).mean(axis=(0, 2))
        top_indices = np.argsort(importance)[-9:][::-1]
        
        x_vals = np.linspace(-2, 2, 100)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, feat_idx in enumerate(top_indices):
            y_vals = []
            for x in x_vals:
                x_tensor = torch.zeros(1, kan_layer.in_features, device=device)
                x_tensor[0, feat_idx] = x
                y = compute_kan_output(kan_layer, x_tensor, output_neuron=0)
                y_vals.append(y[0])
            
            axes[idx].plot(x_vals, y_vals, 'b-', linewidth=2)
            axes[idx].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            axes[idx].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            axes[idx].set_title(f'Neuron #{feat_idx}', fontsize=10)
            axes[idx].set_xlabel('Input value', fontsize=8)
            axes[idx].set_ylabel('Output', fontsize=8)
            axes[idx].grid(True, alpha=0.3)
        
        for idx in range(len(top_indices), 9):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'KAN-Transformer Layer {layer_idx+1}: FFN Splines {suffix}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'kan_transformer_ffn_layer{layer_idx+1}{suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: kan_transformer_ffn_layer{layer_idx+1}{suffix}.png")

# ============== TRAINING FUNCTION ==============

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, gpu_num=0, task='binary', output_dim=2):
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for cat, cont, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            cat, cont, target = cat.to(device), cont.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(cat, cont)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        y_true = []
        preds = []
        probs_all = []
        
        with torch.no_grad():
            for cat, cont, target in val_loader:
                cat, cont, target = cat.to(device), cont.to(device), target.to(device)
                output = model(cat, cont)
                if isinstance(output, tuple):
                    output = output[0]
                probs = torch.softmax(output, dim=1)
                if output_dim == 2:
                    probs_all.extend(probs[:, 1].cpu().numpy())
                else:
                    probs_all.extend(probs.cpu().numpy())
                preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                y_true.extend(target.cpu().numpy())
        
        if output_dim == 2:
            val_metric = roc_auc_score(y_true, probs_all) if len(np.unique(y_true)) > 1 else 0.5
        else:
            val_metric = f1_score(y_true, preds, average='macro')
        
        scheduler.step(val_metric)
        print(f"Epoch {epoch+1}: Val Metric={val_metric:.4f}")
    
    return model

# ============== HELPER FUNCTIONS ==============

def get_quantile_bins(x_cont, n_bins=4):
    if x_cont.ndim != 2:
        raise ValueError("x_cont must be a 2D tensor")
    feature_dim = x_cont.shape[1]
    bins = torch.zeros(feature_dim, n_bins + 1, device=x_cont.device)
    for i in range(feature_dim):
        quantiles = torch.quantile(x_cont[:, i], torch.linspace(0, 1, n_bins + 1, device=x_cont.device), dim=0)
        bins[i] = quantiles
    return bins

def build_vocabulary(dataset, continuous_features, categorical_features, target_name):
    all_vocabularies = []
    for fold in range(1, 2):
        paths = get_data_path(dataset, fold)
        train_df = pd.read_csv(paths['train']).fillna('EMPTY')
        temp_dataset = TabularDataset(train_df, target_name, 'classification', 
                                       categorical_features, continuous_features)
        all_vocabularies.append(temp_dataset.get_vocabulary())
    
    combined = {}
    for vocab in all_vocabularies:
        for col, mapping in vocab.items():
            if col not in combined:
                combined[col] = mapping
            else:
                combined[col].update(mapping)
    
    final = {}
    for col in combined:
        unique_vals = sorted(str(v) for v in combined[col].keys())
        final[col] = {v: i for i, v in enumerate(unique_vals)}
    return final

def create_model(vocabulary, continuous_features, categorical_features, device, output_dim=2):
    bins = torch.zeros(len(continuous_features), 5, device=device)
    model = TabKANet(
        output_dim=output_dim,
        vocabulary=vocabulary,
        num_continuous_features=len(continuous_features),
        embedding_dim=64,
        nhead=8,
        num_layers=3,
        dim_feedforward=128,
        attn_dropout_rate=0.1,
        mlp_hidden_dims=[32],
        activation='relu',
        ffn_dropout_rate=0.1,
        learninable_noise=False,
        bins=bins,
        classifier_type='mlp',
        transformer_type='kan',
        return_attention=False
    )
    return model

# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bankmarketing', choices=['bankmarketing', 'multi_forest'])
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--gpunum', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pruning_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    config = DATASET_CONFIG[args.dataset]
    continuous_features = config['continuous']
    cont_names = config['cont_names']
    categorical_features = config['categorical']
    target_name = config['target']
    output_dim = config['output_dim']
    task = config['task']
    
    device = torch.device(f"cuda:{args.gpunum}" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print(f"TRAIN → PRUNE → VISUALIZE")
    print(f"Dataset: {args.dataset}, Fold: {args.fold}")
    print(f"Epochs: {args.epochs}, Pruning ratio: {args.pruning_ratio*100}%")
    print("="*80)
    
    # Load data
    paths = get_data_path(args.dataset, args.fold)
    train_data = pd.read_csv(paths['train']).fillna('EMPTY')
    val_data = pd.read_csv(paths['val']).fillna('EMPTY')
    test_data = pd.read_csv(paths['test']).fillna('EMPTY')
    
    vocabulary = build_vocabulary(args.dataset, continuous_features, categorical_features, target_name)
    
    train_dataset, test_dataset, val_dataset = get_dataset(
        train_data, test_data, val_data, target_name,
        'classification', categorical_features, continuous_features)
    
    train_loader, test_loader, val_loader = get_data_loader(
        train_dataset, test_dataset, val_dataset,
        train_batch_size=128, inference_batch_size=128)
    
    # Bins
    data_numpy = {'train': {'x_cont': train_dataset.continuous_data}}
    data = {part: {k: torch.as_tensor(v, device=device).float() for k, v in data_numpy[part].items()}
            for part in data_numpy}
    bins = get_quantile_bins(data['train']['x_cont'], n_bins=4)
    
    # Create model
    model = create_model(vocabulary, continuous_features, categorical_features, device, output_dim)
    model.encoders['continuous_feature_encoder'].bins = bins
    model.to(device)
    
    # ============ 1. TRAIN ============
    print("\n" + "-"*40)
    print("STEP 1: TRAINING")
    print("-"*40)
    model = train_model(model, train_loader, val_loader, args.epochs, gpu_num=args.gpunum, task=task, output_dim=output_dim)
    
    # Evaluate
    model.eval()
    y_true = []
    preds = []
    probs_all = []
    
    with torch.no_grad():
        for cat, cont, target in test_loader:
            cat, cont, target = cat.to(device), cont.to(device), target.to(device)
            output = model(cat, cont)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)
            if output_dim == 2:
                probs_all.extend(probs[:, 1].cpu().numpy())
            else:
                probs_all.extend(probs.cpu().numpy())
            preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            y_true.extend(target.cpu().numpy())
    
    if output_dim == 2:
        test_metric_before = roc_auc_score(y_true, probs_all) if len(np.unique(y_true)) > 1 else 0.5
        metric_name = "AUC"
    else:
        test_metric_before = f1_score(y_true, preds, average='macro')
        metric_name = "F1"
    
    print(f"Test {metric_name} before pruning: {test_metric_before:.4f}")
    
    # ============ 2. VISUALIZE BEFORE PRUNING ============
    print("\n" + "-"*40)
    print("STEP 2: VISUALIZE BEFORE PRUNING")
    print("-"*40)
    visualize_encoder_kan(model, continuous_features, cont_names, suffix='_before_pruning')
    visualize_kan_transformer_ffn(model, suffix='_before_pruning')
    
    # ============ 3. PRUNE ============
    print("\n" + "-"*40)
    print(f"STEP 3: PRUNING ({args.pruning_ratio*100}%)")
    print("-"*40)
    model = prune_kan_transformer(model, pruning_ratio=args.pruning_ratio)
    
    # Evaluate after pruning
    model.eval()
    y_true = []
    preds = []
    probs_all = []
    
    with torch.no_grad():
        for cat, cont, target in test_loader:
            cat, cont, target = cat.to(device), cont.to(device), target.to(device)
            output = model(cat, cont)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)
            if output_dim == 2:
                probs_all.extend(probs[:, 1].cpu().numpy())
            else:
                probs_all.extend(probs.cpu().numpy())
            preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            y_true.extend(target.cpu().numpy())
    
    if output_dim == 2:
        test_metric_after = roc_auc_score(y_true, probs_all) if len(np.unique(y_true)) > 1 else 0.5
    else:
        test_metric_after = f1_score(y_true, preds, average='macro')
    
    print(f"Test {metric_name} after pruning: {test_metric_after:.4f}")
    print(f"Change: {test_metric_after - test_metric_before:.4f}")
    
    # ============ 4. VISUALIZE AFTER PRUNING ============
    print("\n" + "-"*40)
    print("STEP 4: VISUALIZE AFTER PRUNING")
    print("-"*40)
    visualize_encoder_kan(model, continuous_features, cont_names, suffix='_after_pruning')
    visualize_kan_transformer_ffn(model, suffix='_after_pruning')
    
    # ============ SUMMARY ============
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test {metric_name} before pruning: {test_metric_before:.4f}")
    print(f"Test {metric_name} after pruning:  {test_metric_after:.4f}")
    print(f"Change: {test_metric_after - test_metric_before:.4f}")
    print("\nFiles saved:")
    print("  - kan_encoder_before_pruning.png")
    print("  - kan_encoder_after_pruning.png")
    print("  - kan_transformer_ffn_layer{1,2,3}_before_pruning.png")
    print("  - kan_transformer_ffn_layer{1,2,3}_after_pruning.png")
    print("="*80)

if __name__ == '__main__':
    main()