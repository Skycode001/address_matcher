import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import generate_features


def generate_training_pairs_simple(df, num_pairs=20000):
    """
    Простая генерация пар для логистической регрессии
    """
    X = []
    y = []

    addresses = df['Адрес'].tolist()

    # Берем случайные адреса
    sample_size = min(len(addresses), 5000)
    addresses_sample = random.sample(addresses, sample_size)

    print(f"Генерация {num_pairs} пар...")

    for i, addr1 in enumerate(tqdm(addresses_sample, desc="Обработка")):
        # Положительные примеры (адрес с самим собой)
        features_self = generate_features(addr1, addr1)
        X.append(features_self)
        y.append(1)

        # Отрицательные примеры (с разными адресами)
        for j in range(min(5, len(addresses_sample))):
            addr2 = addresses_sample[(i + j + 1) % len(addresses_sample)]
            if addr1 != addr2:
                features_diff = generate_features(addr1, addr2)
                X.append(features_diff)
                y.append(0)

        if len(X) >= num_pairs:
            break

    return np.array(X[:num_pairs]), np.array(y[:num_pairs])

def main():
    print("Загрузка базы адресов...")
    df = pd.read_csv('data/addresses.csv')
    print(f"Загружено {len(df)} адресов")

    print("Генерация обучающих пар...")
    X, y = generate_training_pairs_simple(df, num_pairs=20000)

    print(f"Создано {len(X)} пар")
    print(f"Положительных: {sum(y)}")
    print(f"Отрицательных: {len(y)-sum(y)}")

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Логистическая регрессия
    print("\nОбучение логистической регрессии...")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=1.0  # Регуляризация
    )
    model.fit(X_train_scaled, y_train)

    # Оценка
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность на тесте: {accuracy:.3f}")

    # Проверка на сложном примере
    print("\nПроверка на примере 'Таллинская 20 корпус 3':")
    test_query = "Таллинская 20 корпус 3"
    test_candidate = "Таллинская улица, дом 20, корпус 3"
    test_features = generate_features(test_query, test_candidate)
    test_scaled = scaler.transform([test_features])
    proba = model.predict_proba(test_scaled)[0][1]
    print(f"  Вероятность совпадения: {proba:.4f}")

    # Сохраняем
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/logistic_regression.pkl', compress=3)
    joblib.dump(scaler, 'models/scaler.pkl', compress=3)
    print("\n✅ Модель сохранена")

if __name__ == "__main__":
    main()
