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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# from config import *
import config
import os

class ModelManager:
    def __init__(self, models_dir=config.MODELS_DIR, metrics_dir=config.METADATA_DIR):
        self.models_dir = models_dir
        self.latest_models_dir = os.path.join(models_dir, "latest")
        self.best_model_dir = os.path.join(models_dir, "best")
        self.metrics_dir = metrics_dir
        self.target = config.TARGET
        self.best_score = -1
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(self.latest_models_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)


    def _get_preprocessor(self, X):
        """Создание пайплайна предобработки"""
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


    def save_model(self, model_name, model, results, best=False):
        """Сохранение модели, обновление симлинков, сохранение метаданных"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{model_name}_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        joblib.dump(model, model_path)

        latest_path = os.path.join(self.latest_models_dir, model_filename)
        os.symlink(model_path, latest_path)

        if best:
            for link in os.listdir(self.best_model_dir):
                best_path = os.path.join(self.best_model_dir, link)
                os.remove(best_path)
            best_path = os.path.join(self.best_model_dir, model_filename)
            os.symlink(model_path, best_path)

        metrics_path = os.path.join(self.metrics_dir, f"train_metrics_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "model": model_name,
                "model_path": model_path,
                "results": results,
                "timestamp": timestamp,
            }, f, indent=4)


    def evaluate(self, name, pipeline, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        score = f1_score(y_test, y_pred)
        accur = accuracy_score(y_test, y_pred)
        results = {
            "f1_score": score,
            "accuracy": accur,
        }
        print(f"--- Модель {name}: F1 = {score:.4f} | Accuracy = {accur:.4f}")
        return results


    def _train_new_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=config.RANDOM_SEED)
        model_candidates = {
            "random_forest": RandomForestClassifier(n_estimators=100,
                                                    max_depth=10,
                                                    random_state=config.RANDOM_SEED),
            "neural_network": MLPClassifier(hidden_layer_sizes=(32, 16),
                                            max_iter=500,
                                            random_state=config.RANDOM_SEED)
        }

        preprocessor = self._get_preprocessor(X)

        for name, model in model_candidates.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            results = self.evaluate(name, pipeline, X_test, y_test)
            score = results['f1_score']
            self.save_model(name, pipeline, results, best=(score > self.best_score))
            best_score = max(self.best_score, score)


    def _update_models(self, X, y):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED)
        return self._train_new_models(X, y)


    def train_and_evaluate(self, data_path):
        """Обучение, валидация и сохранение метаданных"""
        print(f"Начало этапа обучения на данных: {data_path}")
        df = pd.read_csv(data_path)

        drop_cols = [self.target, 'INSR_BEGIN', 'INSR_END']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        cat_cols = X.select_dtypes(include=['object', 'bool']).columns
        for col in cat_cols:
            X[col] = X[col].astype(str)
        y = df[self.target]

        if os.listdir(self.latest_models_dir):
            self._update_models(X, y)
        else:
            self._train_new_models(X, y)



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
