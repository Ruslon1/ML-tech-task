# Дообучение Mistral через QLoRA под стиль customer support

## Обзор

В этом проекте реализован полный цикл instruction-tuning:

1. подготовка небольшого доменного датасета,
2. дообучение открытой LLM через QLoRA,
3. сравнение базовой и дообученной модели на отдельном evaluation set.

## Постановка задачи

- Базовая модель: `mistralai/Mistral-7B-Instruct-v0.2`
- Метод дообучения: `QLoRA`
- Стек обучения: `Transformers + PEFT`
- Логирование: `MLflow`

## Структура репозитория

- [prepare_data.py](/Users/ruslan/Projects/ML%20tech%20task/prepare_data.py) — подготовка финального train датасета
- [train.py](/Users/ruslan/Projects/ML%20tech%20task/train.py) — QLoRA fine-tuning
- [tracking.py](/Users/ruslan/Projects/ML%20tech%20task/tracking.py) — логирование в MLflow
- [eval.py](/Users/ruslan/Projects/ML%20tech%20task/eval.py) — сравнение базовой и затюненой модели
- [data/dataset.jsonl](/Users/ruslan/Projects/ML%20tech%20task/data/dataset.jsonl) — финальный train dataset
- [data/evaluation.jsonl](/Users/ruslan/Projects/ML%20tech%20task/data/evaluation.jsonl) — evaluation prompts
- [EDA.ipynb](/Users/ruslan/Projects/ML%20tech%20task/EDA.ipynb) — EDA
- [colab.ipynb](/Users/ruslan/Projects/ML%20tech%20task/colab.ipynb) — для взаимодействия с Colab

## Шаг 1. Данные

### Источник данных

Использовал датасет с Hugging Face:

- `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- [Датасет](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

Взял из датасета только одну категорию, чтобы уменьшить его (а то обучение на T4 было бы сильно дольше):

- `category == "CANCEL"`

### Финальный train dataset

- Финальный файл: [data/dataset.jsonl](/Users/ruslan/Projects/ML%20tech%20task/data/dataset.jsonl)
- Финальный размер: `950` примеров
- Формат: JSONL с полями:
  - `instruction`
  - `response`

### Предобработка

Предобработка реализована в [prepare_data.py](/Users/ruslan/Projects/ML%20tech%20task/prepare_data.py).

Что сделал:

- оставил только категорию `CANCEL`
- сохранил только поля `instruction` и `response`
- удалил пустые строки
- нормализовал пробелы
- удалил дубликаты

Очистку намерненно оставил простой. Датасет уже был супер чистым, я просмотрел его весь глазами, поэтому для этой задачи хватило легкой rule-based подготовки.

### EDA

Исследование данных находится в [EDA.ipynb](/Users/ruslan/Projects/ML%20tech%20task/EDA.ipynb):

- распределение категорий
- проверка на пропуски
- проверка на дубликаты
- статистика длин
- ручной просмотр примеров

## Шаг 2. Fine-Tuning

### Метод обучения

Модель дообучалась через QLoRA:

- базовая модель загружается в 4-bit
- обучаются только LoRA adapters
- сохраняются только adapter weights

Такая сильная квантизация опять же из-за плохой доступности железа мне

### Конфигурация обучения

Основные настройки находятся в [train.py](/Users/ruslan/Projects/ML%20tech%20task/train.py):

- модель: `mistralai/Mistral-7B-Instruct-v0.2`
- max sequence length: `256`
- learning rate: `2e-4`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- optimizer: `paged_adamw_8bit`

Перед обучением каждая пара `instruction/response` приводится к формату:

```text
<s>[INST] instruction [/INST] response</s>
```

### Артефакты обучения

- финальный adapter: `outputs/mistral-lora/adapter`
- training checkpoints: `outputs/mistral-lora/checkpoint-*`
- MLflow artifacts: `mlruns/`

### Логирование

Логирование реализовано в [tracking.py](/Users/ruslan/Projects/ML%20tech%20task/tracking.py).

Логируются:

- основные гиперпараметры
- training loss из `trainer.state.log_history`
- артефакты обучения

График training loss:

![Training Loss](/loss.png)

## Шаг 3. Оценка

### Evaluation set

Evaluation set находится в [data/evaluation.jsonl](/Users/ruslan/Projects/ML%20tech%20task/data/evaluation.jsonl).
Его я сгенерировал синтетически, через ChatGPT
- размер: `15` примеров
- формат:
  - `instruction`
  - `response` как reference answer

Примеры остались на том же домене.

### Метрики

В [eval.py](/Users/ruslan/Projects/ML%20tech%20task/eval.py) сравниваются:

- ответы базовой модели
- ответы дообученной модели

с reference responses по метрикам:

- `ROUGE-L`
- `BERTScore F1`

Подробные результаты сохраняются в:

- [outputs/eval_results.csv](/Users/ruslan/Projects/ML%20tech%20task/outputs/eval_results.csv)

### Качественные выводы

По примерам из `eval_results.csv` видно, что дообученная модель стала лучше соответствовать целевому support-style.

Что стало лучше:

- ответы стали более похожи на customer-support реплики (фразы по типу 'I'll get right on it!' в начале респонса)
- tuned model чаще отправляет пользователя в billing / contract / fees sections
- ответы стали более согласованными внутри узкого cancellation domain
- в целом формулировки стали ближе к target style из train set

Что стало хуже:

- ответы стали более шаблонными
- местами заметна повторяемость формулировок
- tuned model стала менее гибкой, чем base model
- из-за узкого и синтетического датасета модель склонна переиспользовать одни и те же support phrases

### Интерпретация результатов

Базовая модель чаще дает более общие и универсальные ответы. Обычно они разумны, но не всегда звучат как специализированный support agent именно для этой задачи.

Дообученная модель стала заметно более специализированной. Она чаще отвечает в ожидаемом customer-support тоне и обычно ближе к желаемому формату.

Хоть к модели и нет претензий — она отлично дообучилась, — есть претензии к данным: они уж слишком синтетические. После всей работы я понял, что стоило выбрать более качественный датасет. Этим я неудовлетворен.