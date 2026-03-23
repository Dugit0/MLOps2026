"""
(a) Обязательная часть:
i. Функционал сбора потоковых данных: разделение исходного набора на
батчи и эмуляция потока (1 балл);
-- Сделано

ii. Разработка хранилища сырых данных: файловая система (1 балл) или
БД (2 балла);
-- Файловая система

iii. Расчет метапараметров: (1-2 балла).
-- Сделано (TODO можно добавить или поменять параметры)

(b) Дополнительные баллы:
i. Создание конфигурационного файла (.py или YAML/JSON/TOML/XML)
с гиперпараметрами сбора (1 балл);
-- config.py

ii. Интеграция с несколькими источниками данных (2 балла);
-- TODO

iii. Система логирования (с использованием библиотек или вручную) и
обработки ошибок при сборе данных (1-2 балла).
-- TODO
"""


import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from config import *


class DataStreamer:
    def __init__(self,
                 source_path=SOURCE_PATH,
                 sort_col=SORT_COLUMN,
                 batch_size=BATCH_SIZE):
        self.source_path = source_path
        self.sort_col = sort_col
        self.batch_size = batch_size
        for d in [RAW_DIR, METADATA_DIR]:
            os.makedirs(d, exist_ok=True)

    def _get_last_index(self):
        """Загружает индекс последней прочитанной строки из файла состояния."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f).get("last_index", 0)
        return 0

    def _save_state(self, new_index):
        """Сохраняет новый индекс после обработки батча."""
        with open(STATE_FILE, "w") as f:
            json.dump({"last_index": new_index, "updated_at": str(datetime.now())}, f)

    def get_next_batch(self):
        """Сортирует данные и отдает следующий батч."""
        df = pd.read_csv(self.source_path)

        df[self.sort_col] = pd.to_datetime(df[self.sort_col])
        df = df.sort_values(by=self.sort_col).reset_index(drop=True)

        start_idx = self._get_last_index()
        end_idx = start_idx + self.batch_size

        if start_idx >= len(df):
            # TODO Как-то грамотнее это обработать
            print("!!! Все данные уже обработаны. Сброс указателя на начало.")
            start_idx = 0
            end_idx = self.batch_size

        batch = df.iloc[start_idx:end_idx].copy()

        # Создаем целевую переменную
        batch['CLAIM_PAID_BINARY'] = (batch['CLAIM_PAID'] > 0).astype(int)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"{RAW_DIR}/batch_{timestamp}.csv"
        batch.to_csv(batch_filename, index=False)

        # Расчет мета-параметров
        meta = {
            "batch_file": batch_filename,
            "row_count": len(batch),
            "start_date": str(batch[self.sort_col].min()),
            "end_date": str(batch[self.sort_col].max()),
            "mean_claim": float(batch['CLAIM_PAID'].mean()) if 'CLAIM_PAID' in batch else 0,
            "target_distribution": batch['CLAIM_PAID_BINARY'].value_counts(normalize=True).to_dict()
        }

        meta_filename = f"{METADATA_DIR}/batch_meta_{timestamp}.json"
        with open(meta_filename, "w") as f:
            json.dump(meta, f, indent=4)

        self._save_state(end_idx)

        print(f"[Stream] Сформирован батч: {batch_filename}")
        print(f"[Stream] Период: {meta['start_date']} - {meta['end_date']}")
        print(f"[Stream] Мета-данные сохранены в: {meta_filename}")

        return batch_filename
