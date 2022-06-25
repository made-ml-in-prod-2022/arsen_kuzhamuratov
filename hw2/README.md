Второе задание: Rest сервис на FastAPI, запуск сервиса в контейнере Docker

Запуск сервиса:
```
python online_inference/app.py
```

Скрипт запроса на сервер:

```
python online_inference/requester.py
```
Запуск Докера:

```
docker build -t online_inference:v1 .
docker run -it -p 8000:8000 online_inference:v1
```
Загрузка образа из docker.hub:
```
docker pull kuzhamuratov/online_inference:v1
docker run -it -p 8000:8000 kuzhamuratov/online_inference:v1
```
