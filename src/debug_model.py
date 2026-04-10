import joblib
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import generate_features, normalize_address

print("=" * 50)
print("ДИАГНОСТИКА МОДЕЛИ")
print("=" * 50)

# 1. Загружаем модель
model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

print(f"\n1. Модель загружена: {type(model).__name__}")
print(f"   Ожидаемое количество признаков: {model.n_features_in_}")

# 2. Тестовые адреса
query = "Таллинская 20 корпус 3"
candidate = "Таллинская улица, дом 20, корпус 3"

print(f"\n2. Тестовые адреса:")
print(f"   Запрос: {query}")
print(f"   Кандидат: {candidate}")

# 3. Генерируем признаки
features = generate_features(query, candidate)
print(f"\n3. Признаки:")
print(f"   Количество признаков: {len(features)}")
print(f"   Значения признаков: {[round(f, 3) for f in features[:5]]}... (первые 5)")

# 4. Масштабируем
features_scaled = scaler.transform([features])
print(f"\n4. После масштабирования:")
print(f"   Форма: {features_scaled.shape}")
print(f"   Значения: {[round(f, 3) for f in features_scaled[0][:5]]}...")

# 5. Предсказание
pred = model.predict(features_scaled)
pred_proba = model.predict_proba(features_scaled)

print(f"\n5. Предсказание:")
print(f"   Класс: {pred[0]}")
print(f"   Вероятности: {pred_proba[0]}")
print(f"   Вероятность класса 1 (совпадение): {pred_proba[0][1]}")

# 6. Проверка на "хорошем" примере (адрес с самим собой)
features_self = generate_features(query, query)
features_self_scaled = scaler.transform([features_self])
pred_self = model.predict(features_self_scaled)
pred_proba_self = model.predict_proba(features_self_scaled)

print(f"\n6. Тот же адрес (сам с собой):")
print(f"   Вероятность совпадения: {pred_proba_self[0][1]}")

# 7. Проверка распределения вероятностей на нескольких примерах
print(f"\n7. Тест на нескольких кандидатах из базы:")
test_candidates = [
    "Таллинская улица, дом 20, корпус 3",
    "Таллинская улица, дом 13, корпус 2", 
    "улица Талалихина, дом 1, корпус 3",
    "ул. Таманская, влд. 2/7/2"
]

for cand in test_candidates:
    feat = generate_features(query, cand)
    feat_scaled = scaler.transform([feat])
    proba = model.predict_proba(feat_scaled)[0][1]
    print(f"   {cand[:40]}... -> ML score: {proba:.4f}")