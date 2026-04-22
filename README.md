
# Traffic Forecast Production-Ready Starter

The project is already configured for real files:

- `data/traffic.csv`
- `data/traffic_year.csv`

Both files can be used simultaneously. During training, dates are automatically combined and duplicates are deleted by the Period field.

## Structure

```text
.
├── api/
│   └── main.py
├── artifacts/
├── data/
│   ├── traffic.csv
│   └── traffic_year.csv
├── train.py
├── inference.py
├── sample_request.json
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## What is `artifacts/`

This folder is empty at first. After training, they are saved there:

- `model.keras' — trained model
- `x_scaler.joblib` — input feature scaler
- `y_scaler.joblib` — the target's scaler
- `metadata.json` — model and metric parameters

## 1. Training

It can be run without arguments at all.:

```bash
python train.py
```

By default, it is used:

- `data/traffic.csv`
- `data/traffic_year.csv`
- `window_size=60`
- `val_len=60`
- `target_column=referral`

If you want explicitly:

```bash
python train.py \
  --train-csvs data/traffic.csv data/traffic_year.csv \
  --artifacts-dir artifacts \
  --window-size 60 \
  --val-len 60 \
  --epochs 30 \
  --batch-size 16 \
  --target-column referral \
  --model-version v1
```

## 2. Local forecast from CSV

After the training:

```bash
python inference.py
```

By default, the inference takes `data/traffic_year.csv` as a recent history.

Or so:

```bash
python inference.py --input-csv data/traffic_year.csv --artifacts-dir artifacts
```

## 3. Launching the API locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API Documentation:
- `http://localhost:8000/docs`

Health check:
- `GET /health`

Forecast:
- `POST /predict`

Example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```

## 4. Docker

### Assembly

```bash
docker build -t traffic-forecast:latest .
```

### Launch

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/data:/app/data \
  traffic-forecast:latest
```

### Via docker compose

```bash
docker compose up --build
```

## Launch order

1. `pip install -r requirements.txt`
2. `python train.py `
3. make sure that 4 files have appeared in `artifacts/`
4. `python inference.py ` or `uvicorn api.main:app ...`

## Important

Until the artifacts are created, the API will not make predictions.  
'/health` at this point will show the status `not_ready'.

---

# Traffic Forecast Production-Ready Starter

Проект уже настроен под реальные файлы:

- `data/traffic.csv`
- `data/traffic_year.csv`

Оба файла можно использовать одновременно. При обучении даты автоматически объединяются и дубликаты удаляются по полю `Период`.

## Структура

```text
.
├── api/
│   └── main.py
├── artifacts/
├── data/
│   ├── traffic.csv
│   └── traffic_year.csv
├── train.py
├── inference.py
├── sample_request.json
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Что такое `artifacts/`

Эта папка сначала пустая. После обучения туда сохраняются:

- `model.keras` — обученная модель
- `x_scaler.joblib` — scaler входных признаков
- `y_scaler.joblib` — scaler таргета
- `metadata.json` — параметры модели и метрики

## 1. Обучение

Можно запускать вообще без аргументов:

```bash
python train.py
```

По умолчанию используется:

- `data/traffic.csv`
- `data/traffic_year.csv`
- `window_size=60`
- `val_len=60`
- `target_column=referral`

Если хотите явно:

```bash
python train.py \
  --train-csvs data/traffic.csv data/traffic_year.csv \
  --artifacts-dir artifacts \
  --window-size 60 \
  --val-len 60 \
  --epochs 30 \
  --batch-size 16 \
  --target-column referral \
  --model-version v1
```

## 2. Локальный прогноз из CSV

После обучения:

```bash
python inference.py
```

По умолчанию инференс берёт `data/traffic_year.csv` как свежую историю.

Или так:

```bash
python inference.py --input-csv data/traffic_year.csv --artifacts-dir artifacts
```

## 3. Запуск API локально

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Документация API:
- `http://localhost:8000/docs`

Проверка состояния:
- `GET /health`

Прогноз:
- `POST /predict`

Пример:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```

## 4. Docker

### Сборка

```bash
docker build -t traffic-forecast:latest .
```

### Запуск

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/data:/app/data \
  traffic-forecast:latest
```

### Через docker compose

```bash
docker compose up --build
```

## Порядок запуска

1. `pip install -r requirements.txt`
2. `python train.py`
3. убедиться, что в `artifacts/` появились 4 файла
4. `python inference.py` или `uvicorn api.main:app ...`

## Важно

Пока артефакты не созданы, API не будет делать прогнозы.  
`/health` в этот момент покажет статус `not_ready`.
