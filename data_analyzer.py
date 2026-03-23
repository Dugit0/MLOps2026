"""
(a) Обязательная часть:
i. Оценка и хранение показателей качества данных (data quality) (1-2
балла);
-- Сделано

ii. Применение методов построения ассоциативных правил (Apriori / FPtree) по бинарным условиям на значения признаков. Выбрать не менее
5 правил для проверки корректности данных (экспертная оценка) или
обогащения признакового пространства (1-3 балла);
-- Apriori

iii. Базовая очистка данных на основе порогов допустимых значений качества (1 балл).
-- Сделано

(b) Дополнительные баллы:
i. Автоматический EDA (1-2 балла);
-- TODO Улучшить
ii. Добавление Feature Engineering (1-2 балла);
-- TODO Генерация более идейных фич

iii. Генерация отчетов о качестве данных (1 балла);
-- TODO

iv. Мониторинг и обработка ситуаций data drift (1-2 балл).
-- TODO
"""


import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from config import *

class DataAnalyzer:
    def __init__(self,
                 metadata_dir=METADATA_DIR,
                 processed_dir=PROCESSED_DIR):
        self.metadata_dir = metadata_dir
        self.processed_dir = processed_dir
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

    def check_completeness(self, df):
        """Оценка полноты данных"""
        df_bin = df.isna()
        total_missings = float(df_bin.sum().sum() / np.prod(df.shape))
        cols_nan_ratio = (df_bin.sum() / df.shape[0]).to_dict()

        return {
            "total_completeness_ratio": 1 - total_missings,
            "column_nan_ratios": cols_nan_ratio
        }

    def check_validity(self, df):
        """Проверка валидности (отрицательные значения в финансовых полях)"""
        # Проверяем столбцы, которые должны быть положительными
        check_cols = ["PREMIUM", "INSURED_VALUE", "CLAIM_PAID", "SEATS_NUM"]
        validity_report = {}
        for col in check_cols:
            if col in df.columns:
                is_valid = not (df[col] < 0).any()
                validity_report[f"valid_{col}"] = bool(is_valid)
        return validity_report

    def check_uniqueness(self, df):
        """Проверка уникальности"""
        duplicates_count = int(df.duplicated().sum())
        return {"duplicates_count": duplicates_count}

    def run_eda_stats(self, df):
        """Автоматический расчет статистик для EDA"""
        stats = {
            "numeric_summary": df.describe().to_dict(),
            "target_drift_check": df["CLAIM_PAID_BINARY"].value_counts(normalize=True).to_dict() if "CLAIM_PAID_BINARY" in df else {}
        }
        return stats

    def feature_engineering(self, df):
        """Генерация признаков"""
        df_new = df.copy()
        # Обработка пола (SEX == 0 -> Юр. лицо)
        if 'SEX' in df_new.columns:
            df_new["is_legal_entity"] = df_new["SEX"] == 0
        return df_new

    def clean_data(self, df):
        """Базовая очистка данных"""
        # Удаляем дубликаты
        df = df.drop_duplicates()

        # Удаляем строки, если больше половины NaN
        limit = df.shape[1] // 2
        df = df.dropna(thresh=limit)

        # Удаляем невалидные строки
        invalid_mask = (
            (df['PREMIUM'] < 0) |
            (df['INSURED_VALUE'] < 0) |
            (df['SEATS_NUM'] < 0)
        )
        df = df[~invalid_mask]

        # Заполняем пропуски в числовых полях медианой (для MVP)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        return df

    def find_association_rules(self, df):
        """
        Поиск ассоциативных правил для проверки корректности данных
        """
        print("Поиск ассоциативных правил...")

        # Подготовка данных для Apriori
        # Выбираем ключевые колонки и превращаем их в бинарные флаги
        subset = pd.DataFrame()

        if 'SEX' in df.columns:
            subset['Is_Legal_Entity'] = (df['SEX'] == 0)
            subset['Is_Male'] = (df['SEX'] == 1)

        if 'PREMIUM' in df.columns:
            # Премия выше медианы
            subset['High_Premium'] = df['PREMIUM'] > df['PREMIUM'].median()

        if 'CLAIM_PAID_BINARY' in df.columns:
            subset['Has_Claim'] = df['CLAIM_PAID_BINARY'] == 1

        if 'TYPE_VEHICLE' in df.columns:
            # Берем топ-3 самых частых типа транспорта
            top_vehicles = df['TYPE_VEHICLE'].value_counts().nlargest(3).index
            for v in top_vehicles:
                subset[f'Vehicle_{v}'] = df['TYPE_VEHICLE'] == v

        # Применяем Apriori
        # min_support — как часто встречается комбинация (10% данных)
        frequent_itemsets = apriori(subset.astype(bool), min_support=0.1, use_colnames=True)

        if frequent_itemsets.empty:
            return []

        # Генерируем правила (confidence — вероятность правила)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        rules = rules.sort_values(by="confidence", ascending=False).head(10)

        formatted_rules = []
        for _, row in rules.iterrows():
            rule_str = f"{list(row['antecedents'])} -> {list(row['consequents'])} (conf: {row['confidence']:.2f})"
            formatted_rules.append(rule_str)

        print(f"Найдено правил: {len(formatted_rules)}")
        return formatted_rules

    def analyze(self, file_path):
        """Запуск полного цикла анализа батча"""
        df = pd.read_csv(file_path)

        quality_report = {}
        quality_report.update(self.check_completeness(df))
        quality_report.update(self.check_validity(df))

        rules = self.find_association_rules(df)
        quality_report["association_rules"] = rules

        quality_report.update(self.run_eda_stats(df))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.metadata_dir}/dq_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(quality_report, f, indent=4)

        df_cleaned = self.clean_data(df)
        df_featured = self.feature_engineering(df_cleaned)

        processed_path = f"{self.processed_dir}/cleaned_{os.path.basename(file_path)}"
        df_featured.to_csv(processed_path, index=False)

        return processed_path, quality_report
