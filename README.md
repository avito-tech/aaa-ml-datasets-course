# aaa-ml-datasets-course
Репозиторий курса по созданию датасетов Академии Аналитиков Авито

## Образ
Билд / перебилд образа
```bash
docker compose -f docker-compose.yaml build
```
Запуск контейнера
```bash
docker compose -f docker-compose.yaml up -d
```
Перезапуск контейнера
```bash
docker compose restart
```

Зайти в запущенный контейнер
```bash
docker compose exec bash
```

## Публикация пакета

```bash
python3 -m pip install --upgrade twine
python3 -m twine upload  dist/*
```

## Использование пакета

```python
pip install aaa-ml-datasets-course
from heuristics.model.classifier import RoomModel
```
