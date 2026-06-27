import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import argparse
import json

from tabkanet.models import TabKANet
from tabkanet.metrics import f1_score_macro
from tabkanet.tools import seed_everything, get_dataset, get_data_loader
from tabkanet.dataset import TabularDataset
from config import get_data_path

# ============== DATASET CONFIGURATION ==============

DATASET_CONFIG = {
    'bankmarketing': {
        'target': 'Output',
        'continuous': ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous'],
        'categorical': ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome'],
        'task': 'binary',
        'output_dim': 2,
        'metric': 'auc'
    },
    'onlineshoper': {
        'target': 'Revenue',
        'continuous': ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues'],
        'categorical': ['Administrative', 'Informational', 'ProductRelated', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'],
        'task': 'binary',
        'output_dim': 2,
        'metric': 'auc'
    },
    'cahouse': {
        'target': 'Output',
        'continuous': ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income'],
        'categorical': ['ocean_proximity'],
        'task': 'regression',
        'output_dim': 1,
        'metric': 'rmse'
    },
    'multi_forest': {
        'target': 'class',
        'continuous': ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'],
        'categorical': ['wilderness_area1', 'wilderness_area2', 'wilderness_area3', 'wilderness_area4', 'soil_type_1', 'soil_type_2', 'soil_type_3', 'soil_type_4', 'soil_type_5', 'soil_type_6', 'soil_type_7', 'soil_type_8', 'soil_type_9', 'soil_type_10', 'soil_type_11', 'soil_type_12', 'soil_type_13', 'soil_type_14', 'soil_type_15', 'soil_type_16', 'soil_type_17', 'soil_type_18', 'soil_type_19', 'soil_type_20', 'soil_type_21', 'soil_type_22', 'soil_type_23', 'soil_type_24', 'soil_type_25', 'soil_type_26', 'soil_type_27', 'soil_type_28', 'soil_type_29', 'soil_type_30', 'soil_type_31', 'soil_type_32', 'soil_type_33', 'soil_type_34', 'soil_type_35', 'soil_type_36', 'soil_type_37', 'soil_type_38', 'soil_type_39', 'soil_type_40'],
        'task': 'multiclass',
        'output_dim': 7,
        'metric': 'f1'
    },
    'multi_seg': {
        'target': 'class',
        'continuous': ['region.centroid.col', 'region.centroid.row', 'region.pixel.count', 'short.line.density.5', 'short.line.density.2', 'vedge.mean', 'vegde.sd', 'hedge.mean', 'hedge.sd', 'intensity.mean', 'rawred.mean', 'rawblue.mean', 'rawgreen.mean', 'exred.mean', 'exblue.mean', 'exgreen.mean', 'value.mean', 'saturation.mean', 'hue.mean'],
        'categorical': [],
        'task': 'multiclass',
        'output_dim': 7,
        'metric': 'f1'
    },
    'sarcos': {
        'target': 'V22',
        'continuous': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'],
        'categorical': [],
        'task': 'regression',
        'output_dim': 1,
        'metric': 'rmse'
    },
    'cpu_small': {
        'target': 'usr',
        'continuous': ['lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'runqsz', 'freemem', 'freeswap'],
        'categorical': [],
        'task': 'regression',
        'output_dim': 1,
        'metric': 'rmse'
    }
}

# ============== PRUNING FUNCTIONS ==============

def prune_kan_layer_simple(layer, pruning_ratio=0.2):
    with torch.no_grad():
        spline_weights = layer.spline_weight
        
        input_importance = torch.abs(spline_weights).mean(dim=(0, 2))
        output_importance = torch.abs(spline_weights).mean(dim=(1, 2))
        
        input_threshold = torch.quantile(input_importance, pruning_ratio)
        output_threshold = torch.quantile(output_importance, pruning_ratio)
        
        input_mask = (input_importance > input_threshold).float()
        output_mask = (output_importance > output_threshold).float()
        
        mask_3d = output_mask[:, None, None] * input_mask[None, :, None]
        layer.spline_weight.data *= mask_3d
        
        mask_2d = output_mask[:, None] * input_mask[None, :]
        layer.base_weight.data *= mask_2d
        
        return (input_mask < 0.5).sum().item(), (output_mask < 0.5).sum().item()

def prune_kan_transformer(model, pruning_ratio=0.2):
    if hasattr(model.transformer_model, 'blocks'):
        for block in model.transformer_model.blocks:
            if hasattr(block, 'kan_ffn'):
                kan_model = block.kan_ffn
                for layer in kan_model.layers:
                    if hasattr(layer, 'spline_weight'):
                        prune_kan_layer_simple(layer, pruning_ratio)
    return model

def prune_standard_transformer(model, pruning_ratio=0.2):
    if hasattr(model.transformer_model, 'transformer'):
        for layer in model.transformer_model.transformer.layers:
            if hasattr(layer, 'linear1'):
                with torch.no_grad():
                    importance = torch.norm(layer.linear1.weight, dim=1)
                    threshold = torch.quantile(importance, pruning_ratio)
                    mask = (importance > threshold).float()
                    layer.linear1.weight.data *= mask[:, None]
                    if layer.linear1.bias is not None:
                        layer.linear1.bias.data *= mask
            if hasattr(layer, 'linear2'):
                with torch.no_grad():
                    importance = torch.norm(layer.linear2.weight, dim=1)
                    threshold = torch.quantile(importance, pruning_ratio)
                    mask = (importance > threshold).float()
                    layer.linear2.weight.data *= mask[:, None]
                    if layer.linear2.bias is not None:
                        layer.linear2.bias.data *= mask
    return model

# ============== METRIC FUNCTIONS ==============

def compute_metrics(y_true, y_pred, y_proba, task, metric_type):
    if task == 'regression':
        mse = np.mean((np.array(y_pred) - np.array(y_true))**2)
        rmse = np.sqrt(mse)
        return -rmse, rmse  # negative for maximization
    
    elif task == 'binary':
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_proba)
            return auc, auc
        else:
            return 0.5, 0.5
    
    elif task == 'multiclass':
        # Macro F1 score
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        return f1, f1  # return f1 as both val and test metric

# ============== TRAINING FUNCTIONS ==============

def train_model(model, train_loader, val_loader, test_loader, epochs, lr=1e-3, gpu_num=0, save_path=None, task='binary', output_dim=2):
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    if task == 'regression':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    best_val_metric = -float('inf')
    best_test_metric = -float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for categorical_data, continuous_data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            categorical_data = categorical_data.to(device)
            continuous_data = continuous_data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(categorical_data, continuous_data)
            if isinstance(output, tuple):
                output = output[0]
            
            if task == 'regression':
                loss = criterion(output.squeeze(), target.float())
            else:
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        y_true = []
        y_pred = []
        y_proba = []
        
        with torch.no_grad():
            for categorical_data, continuous_data, target in val_loader:
                categorical_data = categorical_data.to(device)
                continuous_data = continuous_data.to(device)
                target = target.to(device)
                
                output = model(categorical_data, continuous_data)
                if isinstance(output, tuple):
                    output = output[0]
                
                if task == 'regression':
                    y_pred.extend(output.squeeze().cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                else:
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                    if output_dim == 2:
                        y_proba.extend(probs[:, 1].cpu().numpy())
                    else:
                        y_proba.extend(probs.cpu().numpy())
        
        val_metric, _ = compute_metrics(y_true, y_pred, y_proba, task, None)
        
        # Test
        y_true = []
        y_pred = []
        y_proba = []
        
        with torch.no_grad():
            for categorical_data, continuous_data, target in test_loader:
                categorical_data = categorical_data.to(device)
                continuous_data = continuous_data.to(device)
                target = target.to(device)
                
                output = model(categorical_data, continuous_data)
                if isinstance(output, tuple):
                    output = output[0]
                
                if task == 'regression':
                    y_pred.extend(output.squeeze().cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                else:
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                    if output_dim == 2:
                        y_proba.extend(probs[:, 1].cpu().numpy())
                    else:
                        y_proba.extend(probs.cpu().numpy())
        
        _, test_metric = compute_metrics(y_true, y_pred, y_proba, task, None)
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        scheduler.step(val_metric)
        
        if task == 'regression':
            print(f"Epoch {epoch+1}: Val RMSE={-val_metric:.4f}, Test RMSE={-test_metric:.4f}")
        elif task == 'binary':
            print(f"Epoch {epoch+1}: Val AUC={val_metric:.4f}, Test AUC={test_metric:.4f}")
        else:
            print(f"Epoch {epoch+1}: Val F1={val_metric:.4f}, Test F1={test_metric:.4f}")
    
    return best_val_metric, best_test_metric

def fine_tune_model(model, train_loader, val_loader, test_loader, epochs=5, lr=1e-4, gpu_num=0, task='binary', output_dim=2):
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    if task == 'regression':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    best_val_metric = -float('inf')
    best_test_metric = -float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for categorical_data, continuous_data, target in tqdm(train_loader, desc=f'Fine-tune Epoch {epoch+1}/{epochs}', leave=False):
            categorical_data = categorical_data.to(device)
            continuous_data = continuous_data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(categorical_data, continuous_data)
            if isinstance(output, tuple):
                output = output[0]
            
            if task == 'regression':
                loss = criterion(output.squeeze(), target.float())
            else:
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        y_true = []
        y_pred = []
        y_proba = []
        
        with torch.no_grad():
            for categorical_data, continuous_data, target in val_loader:
                categorical_data = categorical_data.to(device)
                continuous_data = continuous_data.to(device)
                target = target.to(device)
                
                output = model(categorical_data, continuous_data)
                if isinstance(output, tuple):
                    output = output[0]
                
                if task == 'regression':
                    y_pred.extend(output.squeeze().cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                else:
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                    if output_dim == 2:
                        y_proba.extend(probs[:, 1].cpu().numpy())
                    else:
                        y_proba.extend(probs.cpu().numpy())
        
        val_metric, _ = compute_metrics(y_true, y_pred, y_proba, task, None)
        
        # Test
        y_true = []
        y_pred = []
        y_proba = []
        
        with torch.no_grad():
            for categorical_data, continuous_data, target in test_loader:
                categorical_data = categorical_data.to(device)
                continuous_data = continuous_data.to(device)
                target = target.to(device)
                
                output = model(categorical_data, continuous_data)
                if isinstance(output, tuple):
                    output = output[0]
                
                if task == 'regression':
                    y_pred.extend(output.squeeze().cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                else:
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                    if output_dim == 2:
                        y_proba.extend(probs[:, 1].cpu().numpy())
                    else:
                        y_proba.extend(probs.cpu().numpy())
        
        _, test_metric = compute_metrics(y_true, y_pred, y_proba, task, None)
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        
        scheduler.step(val_metric)
        
        if task == 'regression':
            print(f"Fine-tune Epoch {epoch+1}: Val RMSE={-val_metric:.4f}, Test RMSE={-test_metric:.4f}")
        elif task == 'binary':
            print(f"Fine-tune Epoch {epoch+1}: Val AUC={val_metric:.4f}, Test AUC={test_metric:.4f}")
        else:
            print(f"Fine-tune Epoch {epoch+1}: Val F1={val_metric:.4f}, Test F1={test_metric:.4f}")
    
    return best_val_metric, best_test_metric

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
    for fold in range(1, 6):
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

def create_kan_model(vocabulary, continuous_features, categorical_features, device, output_dim=2):
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

def create_standard_model(vocabulary, continuous_features, categorical_features, device, output_dim=2):
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
        transformer_type='standard',
        return_attention=False
    )
    return model

def run_fold(fold, dataset_config, dataset_name, args, device):
    """Run single fold experiment"""
    
    target_name = dataset_config['target']
    continuous_features = dataset_config['continuous']
    categorical_features = dataset_config['categorical']
    task = dataset_config['task']
    output_dim = dataset_config['output_dim']
    
    # Load data
    paths = get_data_path(dataset_name, fold)
    train_data = pd.read_csv(paths['train']).fillna('EMPTY')
    val_data = pd.read_csv(paths['val']).fillna('EMPTY')
    test_data = pd.read_csv(paths['test']).fillna('EMPTY')
    
    # Build vocabulary
    vocabulary = build_vocabulary(dataset_name, continuous_features, categorical_features, target_name)
    
    # Create datasets and loaders
    if task in ['binary', 'multiclass']:
        target_dtype = 'classification'
    else:
        target_dtype = 'regression'

    train_dataset, test_dataset, val_dataset = get_dataset(
        train_data, test_data, val_data, target_name,
        target_dtype, categorical_features, continuous_features)
    
    train_loader, test_loader, val_loader = get_data_loader(
        train_dataset, test_dataset, val_dataset,
        train_batch_size=128, inference_batch_size=128)
    
    # Calculate bins
    data_numpy = {'train': {'x_cont': train_dataset.continuous_data}}
    data = {part: {k: torch.as_tensor(v, device=device).float() for k, v in data_numpy[part].items()}
            for part in data_numpy}
    bins = get_quantile_bins(data['train']['x_cont'], n_bins=4) if len(continuous_features) > 0 else None
    
    # KAN Model
    model_kan = create_kan_model(vocabulary, continuous_features, categorical_features, device, output_dim)
    if bins is not None:
        model_kan.encoders['continuous_feature_encoder'].bins = bins
    model_kan.to(device)
    
    val_metric, test_metric = train_model(
        model_kan, train_loader, val_loader, test_loader, epochs=10, 
        lr=1e-3, gpu_num=args.gpunum, save_path=f'kan_model_fold{fold}.pth',
        task=task, output_dim=output_dim
    )
    kan_original = test_metric
    
    # KAN Pruning + Fine-tune
    model_kan_pruned = create_kan_model(vocabulary, continuous_features, categorical_features, device, output_dim)
    model_kan_pruned.load_state_dict(torch.load(f'kan_model_fold{fold}.pth'))
    if bins is not None:
        model_kan_pruned.encoders['continuous_feature_encoder'].bins = bins
    model_kan_pruned.to(device)
    
    model_kan_pruned = prune_kan_transformer(model_kan_pruned, pruning_ratio=args.pruning_ratio)
    model_kan_pruned.to(device)
    
    # Evaluate after pruning
    model_kan_pruned.eval()
    y_true = []
    y_pred = []
    y_proba = []
    
    with torch.no_grad():
        for cat, cont, target in test_loader:
            cat, cont, target = cat.to(device), cont.to(device), target.to(device)
            out = model_kan_pruned(cat, cont)
            if isinstance(out, tuple):
                out = out[0]
            
            if task == 'regression':
                y_pred.extend(out.squeeze().cpu().numpy())
                y_true.extend(target.cpu().numpy())
            else:
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(out, dim=1)
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(target.cpu().numpy())
                if output_dim == 2:
                    y_proba.extend(probs[:, 1].cpu().numpy())
                else:
                    y_proba.extend(probs.cpu().numpy())
    
    _, kan_pruned = compute_metrics(y_true, y_pred, y_proba, task, None)
    
    val_metric, test_metric = fine_tune_model(
        model_kan_pruned, train_loader, val_loader, test_loader, epochs=5, 
        lr=1e-4, gpu_num=args.gpunum, task=task, output_dim=output_dim
    )
    kan_finetuned = test_metric
    
    # Standard Model
    model_std = create_standard_model(vocabulary, continuous_features, categorical_features, device, output_dim)
    if bins is not None:
        model_std.encoders['continuous_feature_encoder'].bins = bins
    model_std.to(device)
    
    val_metric, test_metric = train_model(
        model_std, train_loader, val_loader, test_loader, epochs=10, 
        lr=1e-3, gpu_num=args.gpunum, save_path=f'standard_model_fold{fold}.pth',
        task=task, output_dim=output_dim
    )
    std_original = test_metric
    
    # Standard Pruning + Fine-tune
    model_std_pruned = create_standard_model(vocabulary, continuous_features, categorical_features, device, output_dim)
    model_std_pruned.load_state_dict(torch.load(f'standard_model_fold{fold}.pth'))
    if bins is not None:
        model_std_pruned.encoders['continuous_feature_encoder'].bins = bins
    model_std_pruned.to(device)
    
    model_std_pruned = prune_standard_transformer(model_std_pruned, pruning_ratio=args.pruning_ratio)
    model_std_pruned.to(device)
    
    # Evaluate after pruning
    model_std_pruned.eval()
    y_true = []
    y_pred = []
    y_proba = []
    
    with torch.no_grad():
        for cat, cont, target in test_loader:
            cat, cont, target = cat.to(device), cont.to(device), target.to(device)
            out = model_std_pruned(cat, cont)
            if isinstance(out, tuple):
                out = out[0]
            
            if task == 'regression':
                y_pred.extend(out.squeeze().cpu().numpy())
                y_true.extend(target.cpu().numpy())
            else:
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(out, dim=1)
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(target.cpu().numpy())
                if output_dim == 2:
                    y_proba.extend(probs[:, 1].cpu().numpy())
                else:
                    y_proba.extend(probs.cpu().numpy())
    
    _, std_pruned = compute_metrics(y_true, y_pred, y_proba, task, None)
    
    val_metric, test_metric = fine_tune_model(
        model_std_pruned, train_loader, val_loader, test_loader, epochs=5, 
        lr=1e-4, gpu_num=args.gpunum, task=task, output_dim=output_dim
    )
    std_finetuned = test_metric
    
    return {
        'fold': fold,
        'kan_original': kan_original,
        'kan_pruned': kan_pruned,
        'kan_finetuned': kan_finetuned,
        'std_original': std_original,
        'std_pruned': std_pruned,
        'std_finetuned': std_finetuned,
        'task': task,
        'metric': dataset_config['metric']
    }

# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bankmarketing', 
                       choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--gpunum', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pruning_ratio', type=float, default=0.2)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    print("="*80)
    print(f"CROSS VALIDATION COMPARISON")
    print(f"Dataset: {args.dataset}")
    print(f"Pruning ratio: {args.pruning_ratio*100}%")
    print(f"Number of folds: {args.folds}")
    print("="*80)
    
    device = torch.device(f"cuda:{args.gpunum}" if torch.cuda.is_available() else "cpu")
    dataset_config = DATASET_CONFIG[args.dataset]
    
    all_results = []
    
    for fold in range(1, args.folds + 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{args.folds}")
        print(f"{'='*60}")
        
        result = run_fold(fold, dataset_config, args.dataset, args, device)
        all_results.append(result)
        
        task = result['task']
        metric_name = "AUC" if task == 'binary' else ("F1" if task == 'multiclass' else "RMSE")
        
        print(f"\nFold {fold} results ({metric_name}):")
        if task == 'regression':
            print(f"  KAN:        Original={-result['kan_original']:.4f}, Pruned={-result['kan_pruned']:.4f}, Fine-tuned={-result['kan_finetuned']:.4f}")
            print(f"  Standard:   Original={-result['std_original']:.4f}, Pruned={-result['std_pruned']:.4f}, Fine-tuned={-result['std_finetuned']:.4f}")
        else:
            print(f"  KAN:        Original={result['kan_original']:.4f}, Pruned={result['kan_pruned']:.4f}, Fine-tuned={result['kan_finetuned']:.4f}")
            print(f"  Standard:   Original={result['std_original']:.4f}, Pruned={result['std_pruned']:.4f}, Fine-tuned={result['std_finetuned']:.4f}")
    
    # Calculate statistics
    kan_original_scores = [r['kan_original'] for r in all_results]
    kan_pruned_scores = [r['kan_pruned'] for r in all_results]
    kan_finetuned_scores = [r['kan_finetuned'] for r in all_results]
    
    std_original_scores = [r['std_original'] for r in all_results]
    std_pruned_scores = [r['std_pruned'] for r in all_results]
    std_finetuned_scores = [r['std_finetuned'] for r in all_results]
    
    task = all_results[0]['task']
    metric_name = "AUC" if task == 'binary' else ("F1" if task == 'multiclass' else "RMSE")
    higher_better = task != 'regression'
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS - {args.dataset.upper()} ({args.folds}-fold cross validation)")
    print("="*80)
    
    print(f"\n{'Model':<35} {'Mean ' + metric_name:<15} {'Std ' + metric_name:<15}")
    print("-" * 65)
    
    if task == 'regression':
        print(f"{'KAN-Transformer (10 epochs)':<35} {-np.mean(kan_original_scores):.4f} +/- {np.std(kan_original_scores):.4f}")
        print(f"{'KAN-Transformer (after pruning)':<35} {-np.mean(kan_pruned_scores):.4f} +/- {np.std(kan_pruned_scores):.4f}")
        print(f"{'KAN-Transformer (pruned + fine-tuned)':<35} {-np.mean(kan_finetuned_scores):.4f} +/- {np.std(kan_finetuned_scores):.4f}")
        print("-" * 65)
        print(f"{'Standard Transformer (10 epochs)':<35} {-np.mean(std_original_scores):.4f} +/- {np.std(std_original_scores):.4f}")
        print(f"{'Standard Transformer (after pruning)':<35} {-np.mean(std_pruned_scores):.4f} +/- {np.std(std_pruned_scores):.4f}")
        print(f"{'Standard Transformer (pruned + fine-tuned)':<35} {-np.mean(std_finetuned_scores):.4f} +/- {np.std(std_finetuned_scores):.4f}")
    else:
        print(f"{'KAN-Transformer (10 epochs)':<35} {np.mean(kan_original_scores):.4f} +/- {np.std(kan_original_scores):.4f}")
        print(f"{'KAN-Transformer (after pruning)':<35} {np.mean(kan_pruned_scores):.4f} +/- {np.std(kan_pruned_scores):.4f}")
        print(f"{'KAN-Transformer (pruned + fine-tuned)':<35} {np.mean(kan_finetuned_scores):.4f} +/- {np.std(kan_finetuned_scores):.4f}")
        print("-" * 65)
        print(f"{'Standard Transformer (10 epochs)':<35} {np.mean(std_original_scores):.4f} +/- {np.std(std_original_scores):.4f}")
        print(f"{'Standard Transformer (after pruning)':<35} {np.mean(std_pruned_scores):.4f} +/- {np.std(std_pruned_scores):.4f}")
        print(f"{'Standard Transformer (pruned + fine-tuned)':<35} {np.mean(std_finetuned_scores):.4f} +/- {np.std(std_finetuned_scores):.4f}")
    
    # Calculate changes
    kan_drop = np.mean([kan_pruned_scores[i] - kan_original_scores[i] for i in range(len(all_results))])
    kan_recovery = np.mean([kan_finetuned_scores[i] - kan_pruned_scores[i] for i in range(len(all_results))])
    kan_final_change = np.mean([kan_finetuned_scores[i] - kan_original_scores[i] for i in range(len(all_results))])
    
    std_drop = np.mean([std_pruned_scores[i] - std_original_scores[i] for i in range(len(all_results))])
    std_recovery = np.mean([std_finetuned_scores[i] - std_pruned_scores[i] for i in range(len(all_results))])
    std_final_change = np.mean([std_finetuned_scores[i] - std_original_scores[i] for i in range(len(all_results))])
    
    print("\n" + "="*80)
    print("CHANGE SUMMARY")
    print("="*80)
    
    direction = "decrease" if task == 'regression' else "increase"
    
    print(f"\nKAN-Transformer:")
    print(f"  Change after pruning: {kan_drop:.4f} +/- {np.std([kan_pruned_scores[i] - kan_original_scores[i] for i in range(len(all_results))]):.4f}")
    print(f"  Change after fine-tune: {kan_recovery:.4f} +/- {np.std([kan_finetuned_scores[i] - kan_pruned_scores[i] for i in range(len(all_results))]):.4f}")
    print(f"  Final change: {kan_final_change:.4f} +/- {np.std([kan_finetuned_scores[i] - kan_original_scores[i] for i in range(len(all_results))]):.4f}")
    
    print(f"\nStandard Transformer:")
    print(f"  Change after pruning: {std_drop:.4f} +/- {np.std([std_pruned_scores[i] - std_original_scores[i] for i in range(len(all_results))]):.4f}")
    print(f"  Change after fine-tune: {std_recovery:.4f} +/- {np.std([std_finetuned_scores[i] - std_pruned_scores[i] for i in range(len(all_results))]):.4f}")
    print(f"  Final change: {std_final_change:.4f} +/- {np.std([std_finetuned_scores[i] - std_original_scores[i] for i in range(len(all_results))]):.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'cv_comparison_{args.dataset}_{timestamp}.json'
    
    output = {
        'dataset': args.dataset,
        'pruning_ratio': args.pruning_ratio,
        'folds': args.folds,
        'task': task,
        'metric': metric_name,
        'results': all_results,
        'statistics': {
            'kan_original': {'mean': float(np.mean(kan_original_scores)), 'std': float(np.std(kan_original_scores))},
            'kan_pruned': {'mean': float(np.mean(kan_pruned_scores)), 'std': float(np.std(kan_pruned_scores))},
            'kan_finetuned': {'mean': float(np.mean(kan_finetuned_scores)), 'std': float(np.std(kan_finetuned_scores))},
            'std_original': {'mean': float(np.mean(std_original_scores)), 'std': float(np.std(std_original_scores))},
            'std_pruned': {'mean': float(np.mean(std_pruned_scores)), 'std': float(np.std(std_pruned_scores))},
            'std_finetuned': {'mean': float(np.mean(std_finetuned_scores)), 'std': float(np.std(std_finetuned_scores))},
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

if __name__ == '__main__':
    main()