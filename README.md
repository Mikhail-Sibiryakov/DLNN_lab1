# Диагностика обучения MLP

Лабораторная работа по бинарной классификации двух зашумлённых полукруглых областей (`make_moons`). Исследуются активации, инициализация весов, режимы оптимизации и глубина нейросети.

[Отчёт](report.pdf) · [Таблицы результатов](results/summary.md)

## Структура репозитория

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── report.pdf                       # итоговый отчёт
├── main.py                          # точка входа
├── data.py                          # данные, разбиение, масштабирование
├── model.py                         # MLP, инициализация, самопроверки
├── training.py                      # обучение, метрики, градиенты
├── experiments.py                   # 12 запусков, выбор модели, test
├── visualization.py                 # графики и таблицы
├── results/
│   ├── data_diagnostics.json        # размеры, классы и статистики признаков
│   ├── model_checks.json            # формы, параметры и градиенты моделей
│   ├── metrics.csv                  # train/validation на эпохах 0–100
│   ├── gradients.csv                # нормы градиентов скрытых весов
│   ├── final_metrics.csv            # четыре метрики 12 запусков на эпохе 100
│   ├── mean_val_loss.csv            # средние validation loss двух seed
│   ├── selection.json               # выбранный запуск и итоговые метрики test
│   ├── selected_model.pt            # веса выбранной модели на эпохе 100
│   └── summary.md                   # таблицы для просмотра
└── figures/
    ├── train_scatter.png
    ├── validation_loss_seed11.png
    ├── validation_accuracy_seed11.png
    ├── gradients_A_seed11.png
    ├── gradients_C_seed11.png
    ├── gradients_D_seed11.png
    └── selected_loss.png
```

## Установка

Проект проверен на Linux с Python 3.12.3. Нужны Python 3.12, модуль `venv` и `pip`, доступный в системном Python. Версии NumPy, SciPy, scikit-learn, PyTorch и Matplotlib зафиксированы в [requirements.txt](requirements.txt).

Из корня проекта:

```bash
python3 -m venv --system-site-packages --without-pip .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
```

Окружение получает доступ к системным пакетам, чтобы использовать уже установленные зависимости без повторного копирования. Параметр `--without-pip` позволяет использовать системный pip без требования `ensurepip`.

## Полный запуск

В активированном окружении, из корня проекта:

```bash
python main.py
```

Программа подготавливает данные и выполняет самопроверки, затем обучает все шесть конфигураций с seed 11 и 29 — всего 12 запусков по 100 эпох. После выбора модели вычисляются метрики test и создаются файлы в `results/` и `figures/`. Повторный запуск перезаписывает результаты и графики.

Вычисления выполняются на **CPU**, с одним потоком PyTorch. Графики сохраняются в PNG без открытия окон.

## Конфигурации и протокол

Все скрытые слои имеют ширину 32; выход — два логита.

| Код | Скрытых слоёв | Активация | Скрытые веса | Оптимизатор | Learning rate |
| --- | ---: | --- | --- | --- | ---: |
| A | 6 | Sigmoid | Xavier normal | Adam | 0.003 |
| B | 6 | Tanh | Xavier normal | Adam | 0.003 |
| C | 6 | ReLU | Xavier normal | Adam | 0.003 |
| D | 6 | ReLU | He normal | Adam | 0.003 |
| E | 6 | ReLU | He normal | SGD | 0.05 |
| F | 1 | ReLU | He normal | Adam | 0.003 |

Данные: `make_moons(n_samples=2400, noise=0.25, random_state=2026)`. Стратифицированное разбиение train/validation/test — 1440/480/480; StandardScaler обучается только на train. Batch size — 128, метрики полных train и validation записываются до обучения и после каждой эпохи. Accuracy хранится долями от 0 до 1.

Для A/C/D градиенты измеряются на первых 128 объектах масштабированного train на эпохах 0, 20 и 100. В `metrics.csv` сохраняются 1212 строк метрик, в `gradients.csv` — 108 норм.

Конфигурация выбирается по минимальному среднему validation loss двух seed на эпохе 100, затем внутри неё выбирается запуск с меньшим validation loss. Точные ничьи разрешаются по букве конфигурации, затем в пользу seed 11. Test оценивается после выбора. В сохранённых результатах выбрана F, seed 29: test loss ≈ 0.131385, accuracy = 0.95.

## Дополнительные режимы

Проверить подготовку данных, формы тензоров, число параметров и градиенты без обучения:

```bash
python main.py --checks-only
```

Заново построить семь графиков и сравнительные таблицы по сохранённым результатам:

```bash
python main.py --plots-only
```

Для этого режима в `results/` должны находиться `metrics.csv`, `gradients.csv`, `final_metrics.csv` и `selection.json`. Обучение и оценка test при нём не повторяются.

Список доступных параметров:

```bash
python main.py --help
```
