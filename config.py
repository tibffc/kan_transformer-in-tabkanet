import os

BASE_PATH = r'C:\Users\user\KAN-vs-MLP\TabKANet'

# Пути к данным
TEMPLATES_PATH = os.path.join(BASE_PATH, 'templates')

# Пути к скриптам
SCRIPTS_PATH = BASE_PATH

def get_data_path(dataset, fold):
    """Возвращает пути к файлам для датасета и фолда"""
    dataset_path = os.path.join(TEMPLATES_PATH, dataset, f'Fold{fold}')
    return {
        'train': os.path.join(dataset_path, 'train.csv'),
        'val': os.path.join(dataset_path, 'val.csv'),
        'test': os.path.join(dataset_path, 'test.csv')
    }