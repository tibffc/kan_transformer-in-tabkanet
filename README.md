# Сравнительный анализ KAN и MLP в архитектуре TabKANet для табличных данных
---

##  Общая информация

Данный репозиторий содержит код и результаты исследовательской работы по **сравнительному анализу Kolmogorov-Arnold Networks (KAN) и Multilayer Perceptrons (MLP)** в архитектуре **TabKANet** для обработки табличных данных.

---

##  Цель исследования

Сравнить эффективность и интерпретируемость двух архитектур:

1. **KAN-Transformer** — замена стандартного MLP в FFN трансформера на KAN
2. **Standard Transformer** — классическая архитектура с MLP в FFN

**Основные вопросы исследования:**
- Как KAN влияет на качество предсказаний?
- Устойчив ли KAN к прунингу (удалению нейронов)?
- Сохраняется ли интерпретируемость после прунинга?

---

## Ключевые результаты

| Модель | Датасет | Метрика | Результат |
|--------|---------|---------|-----------|
| **KAN-Transformer** | BankMarketing | AUC | **0.9345 ± 0.0010** |
| Standard Transformer | BankMarketing | AUC | 0.9327 ± 0.0020 |
| KAN-Transformer (после прунинга 20%) | BankMarketing | AUC | 0.9334 ± 0.0010 |
| KAN-Transformer | ForestCovertype | Macro F1 | **0.6915 ± 0.0054** |
| Standard Transformer | ForestCovertype | Macro F1 | 0.6776 ± 0.0046 |

**Выводы:**
- KAN-Transformer стабильнее (меньшее стандартное отклонение)
- Прунинг 20% практически не влияет на качество
- KAN обеспечивает интерпретируемость через визуализацию сплайнов
- На сложных датасетах KAN показывает лучшее восстановление после прунинга

---
Датасеты должны быть в папке templates/ в следующей структуре:

templates/
├── bankmarketing/
│   ├── Fold1/train.csv, val.csv, test.csv
│   ├── ...
│   └── Fold5/
├── onlineshoper/
│   └── ...
└── multi_forest/
    └── ...

Сравнение KAN vs MLP с прунингом:
python compare_models.py --dataset bankmarketing --pruning_ratio 0.2

Обучение + прунинг + визуализация:
python train_prune_visualize.py --dataset bankmarketing --epochs 10 --pruning_ratio 0.2
