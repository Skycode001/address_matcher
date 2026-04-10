"""
Быстрый тест без обучения модели
Просто для проверки работы нормализации и нечеткого поиска
"""

import pandas as pd
from rapidfuzz import process, fuzz
import sys
import os

# Добавляем путь к корневой папке проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import normalize_address

def quick_search(df, query):
    """
    Быстрый поиск без ML модели
    """
    query_norm = normalize_address(query)
    addresses_norm = [normalize_address(addr) for addr in df['Адрес']]
    
    results = process.extract(query_norm, addresses_norm, scorer=fuzz.ratio, limit=3)
    
    print(f"\nПоиск: '{query}'")
    print("-" * 50)
    
    for addr_norm, score, idx in results:
        original_addr = df.iloc[idx]['Адрес']
        unom = df.iloc[idx]['УНОМ']
        print(f"УНОМ: {unom} | {original_addr} (совпадение: {score}%)")
    
    return results

def main():
    # Загружаем базу адресов
    print("Загрузка базы адресов...")
    df = pd.read_csv('data/addresses.csv')
    print(f"Загружено {len(df)} адресов")
    
    print("\n" + "="*50)
    print("Быстрый тест (без ML модели)")
    print("Введите 'exit' для выхода")
    print("="*50)
    
    while True:
        query = input("\nВведите адрес: ").strip()
        if query.lower() == 'exit':
            break
        if query:
            quick_search(df, query)

if __name__ == "__main__":
    main()