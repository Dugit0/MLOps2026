"""
3.
(b) Дополнительные баллы:
i. Создание нескольких вариантов предобработки с дальнейшим перебором при поиске лучшей модели (1-2 балла).
-- TODO

4. Обучение/дообучение модели (входит в pipeline построения модели)
(1–5 баллов):
(a) Обязательная часть: построение не менее 2 моделей (обязательно реализовать деревья решений (либо ансамбли, построенные на деревьях)
и нейронные сети) (1 балл).
-- Сделано

(b) Дополнительные баллы:
i. Реализация дообучения предыдущей модели (без обучения модели с
нуля) (1-2 балла);
-- TODO

ii. Разработка нескольких моделей с различной устойчивостью к входным
данным (1-2 балла).
-- TODO

5. Валидация модели (2–10 баллов):
(a) Обязательная часть:
i. Оценка качества модели/моделей (hold-out/CV/TimeSeriesCV) (1-3
балла);
-- Сделано

ii. Разработка хранилища версий моделей и контроль качества (1-2 балла).
(b) Дополнительные баллы:
i. Интерпретация прогнозов (визуализация структуры дерева, оценка коэффициентов LR, демонстрация ближайших соседей, LIME, SHAP) (1-
3 балла);
-- TODO

ii. Мониторинг и обработка ситуаций model drift (1–2 балла).
-- TODO

6. Обслуживание модели (1–6 баллов):
(a) Обязательная часть: выбор и упаковка (сериализация) финальной модели
(или нескольких) (1-2 балла);
-- Сделано

(b) Дополнительные баллы:
i. Мониторинг производительности (времени применения/памяти) (1-2 балла);
-- TODO

ii. Обеспечение гибкого прогноза на основе данных (выбор модели при
разреженных данных или аномальных значениях) (1-2 балла).
-- TODO
"""


import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import *

class ModelManager:
    def __init__(self, models_dir=MODELS_DIR, metrics_dir=METADATA_DIR):
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.target = TARGET
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

    def _get_preprocessor(self, X):
        """Создание пайплайна предобработки (Пункт 3.a)"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])
        return preprocessor

    def train_and_evaluate(self, data_path):
        """Обучение, валидация и выбор лучшей модели"""
        print(f"Начало этапа обучения на данных: {data_path}")
        df = pd.read_csv(data_path)

        # Подготовка X и y
        drop_cols = [self.target, 'INSR_BEGIN', 'INSR_END']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # ---------
        cat_cols = X.select_dtypes(include=['object', 'bool']).columns
        for col in cat_cols:
            X[col] = X[col].astype(str)
        # ---------
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        model_candidates = {
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED),
            "neural_network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=RANDOM_SEED)
        }

        best_score = -1
        best_model = None
        best_model_name = ""
        results = {}

        preprocessor = self._get_preprocessor(X)

        for name, model in model_candidates.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            score = f1_score(y_test, y_pred)
            accur = accuracy_score(y_test, y_pred)

            results[name] = {
                "f1_score": score,
                "accuracy": accuracy_score(y_test, y_pred)
            }

            print(f"--- Модель {name}: F1 = {score:.4f} | Accuracy = {accur:.4f}")

            if score > best_score:
                best_score = score
                best_model = pipeline
                best_model_name = name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{best_model_name}_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)

        joblib.dump(best_model, model_path)

        # Сохраняем ярлык на последнюю лучшую модель
        joblib.dump(best_model, os.path.join(self.models_dir, "best_model.pkl"))

        # Сохраняем метрики
        metrics_path = os.path.join(self.metrics_dir, f"train_metrics_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "best_model": best_model_name,
                "all_results": results,
                "timestamp": timestamp,
                "data_used": data_path
            }, f, indent=4)

        print(f"Лучшая модель ({best_model_name}) сохранена: {model_path}")
        return model_path

    def predict(self, input_data_path):
        """Инференс (применение модели)"""
        best_model_path = os.path.join(self.models_dir, "best_model.pkl")
        if not os.path.exists(best_model_path):
            return "Ошибка: Модель не найдена. Сначала запустите обучение (update)."

        model = joblib.load(best_model_path)
        df = pd.read_csv(input_data_path)

        preds = model.predict(df)
        df['predict'] = preds

        output_path = f"data/processed/predictions_{datetime.now().strftime('%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        return output_path
