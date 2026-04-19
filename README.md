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
