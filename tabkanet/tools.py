import os
import logging
from typing import Optional, Callable, Tuple, Literal, Union, Dict, List
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tabkanet.dataset import TabularDataset

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_data(data_path: str, split_val: bool=True, 
             val_params: Optional[Dict[str, Union[float, int]]]={'test_size': 0.05, 'random_state': None},
             index_col: Optional[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if index_col is not None:
        train_data = pd.read_csv(os.path.join(data_path, 'train.csv'), index_col=index_col)
        test_data = pd.read_csv(os.path.join(data_path, 'test.csv'), index_col=index_col)
    else:
        train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    
    if split_val:
        if val_params is None:
            raise ValueError('val_params must be provided if split_val is True')
        train_data, val_data = train_test_split(train_data, **val_params)
    else:
        val_data = None
    
    return train_data, test_data, val_data

def get_dataset(train_data: pd.DataFrame, test_data: pd.DataFrame, val_data: Optional[pd.DataFrame],
                    target_name: str, target_dtype: Union[Literal['regression', 'classification'], torch.dtype],
                    categorical_features: Optional[List[str]], continuous_features: Optional[List[str]]) \
                        -> Tuple[TabularDataset, TabularDataset, TabularDataset]:
    train_dataset = TabularDataset(train_data, target_name, target_dtype, categorical_features, continuous_features)
    val_dataset = TabularDataset(val_data, target_name, target_dtype, categorical_features, continuous_features)
    test_dataset = TabularDataset(test_data, target_name, target_dtype, categorical_features, continuous_features)
    return train_dataset, test_dataset, val_dataset

def get_data_loader(train_dataset, test_dataset, val_dataset, 
                    train_batch_size: int, inference_batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=inference_batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, num_workers=4, shuffle=False)
    return train_loader, test_loader, val_loader

def train(model: torch.nn.Module, epochs: int, task: Literal['regression', 'classification'],
          train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
          optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss, 
          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
          custom_metric: Optional[Callable[[Tuple[np.ndarray, np.ndarray]], float]] = None, 
          maximize: bool = False, scheduler_custom_metric: bool = False, 
          early_stopping_patience: int = 5, early_stopping_start_from: int = 0, gpu_num: int = 1,
          save_model_path: Optional[str] = None, save_attentions: bool = False) -> Tuple[List, List, List]:
    
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device: {device}')

    best_metric = float('inf') if not maximize else float('-inf')
    best_model_params = None
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []
    best_auc = 0.0
    
    # Для сохранения attention весов
    if save_attentions:
        model.saved_attentions = []

    early_stopping_counter = 0

    model.train()
    model.to(device)
    
    for epoch in tqdm(range(epochs), desc='Epochs'):
        total_loss = 0
        train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (categorical_data, continuous_data, target) in train_loader_tqdm:
            categorical_data = categorical_data.to(device)
            continuous_data = continuous_data.to(device)
            if task == 'regression':
                target = target.unsqueeze(1)
            target = target.to(device)
            optimizer.zero_grad()

            # Forward pass с поддержкой attention
            result = model(categorical_data, continuous_data)
            if isinstance(result, tuple):
                output, attentions = result
                if save_attentions and hasattr(model, 'saved_attentions'):
                    model.saved_attentions.append(attentions)
            else:
                output = result
            
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loader_tqdm.set_postfix(loss=total_loss / (batch_idx + 1))
        
        train_loss = total_loss / len(train_loader)
        train_loss_history.append(train_loss)

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0
            y_true = []
            y_pred = []
            predictions = []

            for categorical_data, continuous_data, target in val_loader:
                categorical_data = categorical_data.to(device)
                continuous_data = continuous_data.to(device)
                if task == 'regression':
                    target = target.unsqueeze(1)
                target = target.to(device)

                result = model(categorical_data, continuous_data)
                if isinstance(result, tuple):
                    output, attentions = result
                    if save_attentions and hasattr(model, 'saved_attentions'):
                        model.saved_attentions.append(attentions)
                else:
                    output = result

                if task == 'regression':
                    y_pred.extend(output.cpu().numpy().reshape(-1).tolist())
                    y_true.extend(target.cpu().numpy().reshape(-1).tolist())
                else:  # classification
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                    predictions.extend(probs[:, 1].cpu().numpy() if output.shape[1] > 1 else probs[:, 0].cpu().numpy())
                
                loss = criterion(output, target)
                val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Метрики
            if task == 'classification':
                if len(np.unique(y_true)) == 2:  # бинарная классификация
                    val_auc = roc_auc_score(y_true, predictions) if len(np.unique(y_true)) > 1 else 0.5
                else:
                    val_auc = roc_auc_score(y_true, predictions, multi_class='ovr') if len(np.unique(y_true)) > 1 else 0.5
                
                val_metric = custom_metric(y_true, y_pred) if custom_metric is not None else val_auc
            else:  # regression
                val_auc = 0
                val_metric = -val_loss if maximize else val_loss
            
            if val_auc > best_auc:
                best_auc = val_auc
                if save_model_path is not None:
                    torch.save(model.state_dict(), save_model_path)
                    logging.info('Model saved')
            
            if scheduler is not None:
                if scheduler_custom_metric:
                    scheduler.step(val_metric)
                else:
                    scheduler.step(val_loss)
            
            val_loss_history.append(val_loss)

        # Test evaluation
        with torch.no_grad():
            model.eval()
            test_loss = 0
            y_true = []
            y_pred = []
            predictions = []
            
            for categorical_data, continuous_data, target in test_loader:
                categorical_data = categorical_data.to(device)
                continuous_data = continuous_data.to(device)
                if task == 'regression':
                    target = target.unsqueeze(1)
                target = target.to(device)

                result = model(categorical_data, continuous_data)
                if isinstance(result, tuple):
                    output, attentions = result
                    if save_attentions and hasattr(model, 'saved_attentions'):
                        model.saved_attentions.append(attentions)
                else:
                    output = result

                if task == 'regression':
                    y_pred.extend(output.cpu().numpy().reshape(-1).tolist())
                    y_true.extend(target.cpu().numpy().reshape(-1).tolist())
                else:
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                    y_pred.extend(preds.cpu().numpy())
                    y_true.extend(target.cpu().numpy())
                    predictions.extend(probs[:, 1].cpu().numpy() if output.shape[1] > 1 else probs[:, 0].cpu().numpy())
                
                loss = criterion(output, target)
                test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_loss_history.append(test_loss)
            
            if task == 'classification':
                test_auc = roc_auc_score(y_true, predictions) if len(np.unique(y_true)) > 1 else 0.5
                test_metric = custom_metric(y_true, y_pred) if custom_metric is not None else test_auc
            else:
                test_auc = 0
                test_metric = -test_loss if maximize else test_loss
        
        # Logging
        if task == 'classification':
            tqdm.write(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Val AUC: {val_auc:.4f}, Val Metric: {val_metric:.4f}, '
                      f'Test AUC: {test_auc:.4f}, Test Metric: {test_metric:.4f}')
        else:
            tqdm.write(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Test Metric: {test_metric:.4f}')
        
        # Early stopping
        if not custom_metric:
            maximize = False
        if not maximize and val_metric < best_metric:
            best_metric = val_metric
            best_model_params = model.state_dict().copy()
            early_stopping_counter = 0
        elif maximize and val_metric > best_metric:
            best_metric = val_metric
            best_model_params = model.state_dict().copy()
            early_stopping_counter = 0
        else:
            if epoch >= early_stopping_start_from:
                early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                tqdm.write('Early stopping')
                break
    
    # Load best model
    if best_model_params is not None:
        model.load_state_dict(best_model_params)
    
    print(f"FINISHED TRAINING, BEST VAL AUC: {best_auc:.4f}")
    
    return train_loss_history, val_loss_history, test_loss_history, best_auc


def inference(model: torch.nn.Module, test_loader: DataLoader, task: Literal['regression', 'classification'] = 'classification') -> np.ndarray:
    if task not in {'regression', 'classification'}:
        raise ValueError(f'Task {task} is not supported yet')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device: {device}')
    model.to(device)

    predictions = []
    y_true = []
    
    with torch.no_grad():
        for categorical_data, continuous_data, target in test_loader:
            categorical_data = categorical_data.to(device)
            continuous_data = continuous_data.to(device)
            
            result = model(categorical_data, continuous_data)
            if isinstance(result, tuple):
                output, _ = result
            else:
                output = result
            
            if task == 'classification':
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                predictions.extend(preds.cpu().numpy())
                y_true.extend(target.numpy())
            else:
                predictions.extend(output.cpu().numpy().reshape(-1).tolist())
    
    return np.array(predictions)