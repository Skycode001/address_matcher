import os
import sys
from collections import defaultdict

import joblib
import pandas as pd
from rapidfuzz import fuzz, process

# Добавляем путь к корневой папке проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import generate_features, normalize_address


class AddressMatcher:
    def __init__(self, addresses_df, use_index=True):
        """
        addresses_df: DataFrame с колонками 'ID', 'Адрес', 'УНОМ'
        use_index: использовать ли хэш-индекс для быстрого поиска
        """
        self.df = addresses_df
        self.addresses = addresses_df['Адрес'].tolist()
        self.unoms = addresses_df['УНОМ'].tolist()
        self.ids = addresses_df['ID'].tolist()
        self.use_index = use_index

        print("Предварительная обработка адресов...")
        # Кэшируем нормализованные адреса
        self.normalized_addresses = []
        for addr in self.addresses:
            try:
                norm_addr = normalize_address(addr)
                self.normalized_addresses.append(norm_addr)
            except Exception as e:
                print(f"Ошибка нормализации адреса: {addr}, ошибка: {e}")
                self.normalized_addresses.append("")

        # Создаем хэш-индекс для мгновенного поиска точных совпадений
        if use_index:
            print("Создание хэш-индекса для быстрого поиска...")
            self.index = defaultdict(list)
            for idx, norm_addr in enumerate(self.normalized_addresses):
                if norm_addr:
                    # Ключ индекса - нормализованный адрес
                    self.index[norm_addr].append(idx)

                    # Также добавляем ключ без номеров домов (для поиска улиц)
                    street_only = ' '.join(norm_addr.split()[:-1]) if len(norm_addr.split()) > 1 else norm_addr
                    if street_only != norm_addr:
                        self.index[street_only].append(idx)

            print(f"Индекс создан: {len(self.index)} уникальных ключей")

        # Загружаем ML модель (логистическая регрессия)
        try:
            self.model = joblib.load('models/logistic_regression.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.model_loaded = True
            print("ML модель (логистическая регрессия) загружена успешно")
        except FileNotFoundError:
            print("Модель не найдена. Запустите python src/train.py для обучения.")
            self.model_loaded = False
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model_loaded = False

    def exact_match_search(self, query_normalized):
        """
        Быстрый поиск по хэш-индексу (O(1) сложность)
        """
        if not self.use_index:
            return None

        # Ищем точное совпадение
        if query_normalized in self.index:
            indices = self.index[query_normalized]
            return [{
                'index': idx,
                'address': self.addresses[idx],
                'unom': self.unoms[idx],
                'id': self.ids[idx],
                'fuzzy_score': 100,
                'exact_match': True
            } for idx in indices]

        # Ищем совпадение по улице без номера дома
        street_only = ' '.join(query_normalized.split()[:-1]) if len(query_normalized.split()) > 1 else query_normalized
        if street_only in self.index:
            indices = self.index[street_only]
            return [{
                'index': idx,
                'address': self.addresses[idx],
                'unom': self.unoms[idx],
                'id': self.ids[idx],
                'fuzzy_score': 90,
                'exact_match': False
            } for idx in indices]

        return None

    def fuzzy_search(self, query_normalized, top_n=20):
        """
        Нечеткий поиск с ограничением по количеству кандидатов
        """
        # Для больших баз используем process.extract с ограничением
        results = process.extract(
            query_normalized,
            self.normalized_addresses,
            scorer=fuzz.ratio,
            limit=top_n,
            score_cutoff=60  # Игнорируем кандидатов с совпадением менее 60%
        )

        candidates = []
        for addr_normalized, score, idx in results:
            candidates.append({
                'index': idx,
                'address': self.addresses[idx],
                'unom': self.unoms[idx],
                'id': self.ids[idx],
                'fuzzy_score': score,
                'exact_match': False
            })

        return candidates

    def find_best_match(self, query, top_n=20):
        """
        Находит лучший адрес по запросу
        """
        query_normalized = normalize_address(query)

        # Шаг 1: Пытаемся найти точное совпадение по индексу (мгновенно)
        exact_matches = self.exact_match_search(query_normalized)

        if exact_matches and len(exact_matches) == 1:
            # Если нашли одно точное совпадение, возвращаем его сразу
            best = exact_matches[0]
            best['final_score'] = 1.0
            best['ml_score'] = 1.0
            return [best]
        elif exact_matches and len(exact_matches) > 1:
            # Если несколько точных совпадений, используем их как кандидатов
            candidates = exact_matches
        else:
            # Шаг 2: Нечеткий поиск (только если нет точных совпадений)
            candidates = self.fuzzy_search(query_normalized, top_n)

        if not candidates:
            return []

        # Шаг 3: Если модель загружена, ранжируем кандидатов с помощью ML
        if self.model_loaded and len(candidates) > 0:
            features_matrix = []
            for candidate in candidates:
                features = generate_features(query, candidate['address'])
                features_matrix.append(features)

            # Масштабируем признаки
            features_matrix_scaled = self.scaler.transform(features_matrix)

            # Получаем вероятности от модели
            probabilities = self.model.predict_proba(features_matrix_scaled)[:, 1]

            # Добавляем ML-оценку к каждому кандидату
            for i, candidate in enumerate(candidates):
                candidate['ml_score'] = probabilities[i]
                # Итоговая оценка (можно настроить веса)
                if candidate.get('exact_match', False):
                    # Для точных совпадений даем максимальный вес
                    candidate['final_score'] = 0.8 + 0.2 * candidate['ml_score']
                else:
                    candidate['final_score'] = 0.6 * candidate['fuzzy_score'] / 100 + 0.4 * candidate['ml_score']

            # Сортируем по финальной оценке
            candidates.sort(key=lambda x: x['final_score'], reverse=True)
        else:
            # Без модели используем только fuzzy_score
            for candidate in candidates:
                candidate['final_score'] = candidate['fuzzy_score'] / 100
                candidate['ml_score'] = candidate['fuzzy_score'] / 100
            candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return candidates

    def search(self, query):
        """
        Основной метод поиска
        """
        print(f"\nПоиск: '{query}'")
        print("-" * 50)

        import time
        start_time = time.time()

        candidates = self.find_best_match(query)

        elapsed_time = (time.time() - start_time) * 1000
        print(f"Время поиска: {elapsed_time:.1f} мс")

        if not candidates:
            print("Ничего не найдено")
            return None

        best = candidates[0]

        # Если уверенность низкая, показываем топ-3
        if best['final_score'] < 0.6 and len(candidates) > 1:
            print("⚠️ Низкая уверенность. Возможные варианты:")
            for i, cand in enumerate(candidates[:3], 1):
                match_type = "✓ Точное" if cand.get('exact_match', False) else "≈ Нечеткое"
                print(f"  {i}. [{match_type}] УНОМ: {cand['unom']} | {cand['address']}")
                print(f"     Уверенность: {cand['final_score']:.2%} (ML: {cand.get('ml_score', 0):.2%})")

            choice = input("\nВыберите номер или нажмите Enter для первого: ")
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                best = candidates[int(choice) - 1]

        print("\n✅ Найден адрес:")
        print(f"   УНОМ: {best['unom']}")
        print(f"   Адрес: {best['address']}")
        if best.get('exact_match', False):
            print("   Тип: Точное совпадение по индексу")
        print(f"   Уверенность: {best['final_score']:.2%}")

        return best

    def search_batch(self, queries):
        """
        Пакетный поиск для множества запросов (быстрее чем по одному)
        """
        results = []
        for query in queries:
            result = self.find_best_match(query)
            if result:
                results.append({
                    'query': query,
                    'unom': result[0]['unom'],
                    'address': result[0]['address'],
                    'score': result[0]['final_score']
                })
        return results

def main():
    # Загружаем базу адресов
    print("Загрузка базы адресов...")
    df = pd.read_csv('data/addresses.csv')
    print(f"Загружено {len(df)} адресов")

    # Проверяем наличие необходимых колонок
    required_columns = ['Адрес', 'УНОМ']
    for col in required_columns:
        if col not in df.columns:
            print(f"Ошибка: В файле отсутствует колонка '{col}'")
            print(f"Доступные колонки: {list(df.columns)}")
            return

    # Создаем матчер с индексом
    matcher = AddressMatcher(df, use_index=True)

    # Интерактивный поиск
    print("\n" + "="*50)
    print("Address Matcher готов к работе (логистическая регрессия)")
    print("Введите 'exit' для выхода, 'batch' для пакетного режима")
    print("="*50)

    while True:
        query = input("\nВведите адрес: ").strip()
        if query.lower() == 'exit':
            break
        elif query.lower() == 'batch':
            print("\nПакетный режим. Введите адреса по одному (пустая строка для завершения):")
            batch_queries = []
            while True:
                q = input("> ").strip()
                if not q:
                    break
                batch_queries.append(q)
            if batch_queries:
                results = matcher.search_batch(batch_queries)
                print("\nРезультаты пакетного поиска:")
                for r in results:
                    print(f"  {r['query']} -> УНОМ: {r['unom']} (уверенность: {r['score']:.2%})")
        elif query:
            matcher.search(query)

if __name__ == "__main__":
    main()
