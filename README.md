# MLOps MVP. Краткое руководство

### Подготовка

Установите зависимости командой
```
pip install -r requirements.txt
```

### Запуск run.py

- update -- `python run.py -mode update` (обучение модели на очередном батче)
- predict -- `python run.py -mode predict -file file` (предсказание целевой переменной)
- summary -- `python run.py -mode summary` (отчет о лучшей модели)
- clear -- `python run.py -mode clear` (очистка данных)
