import os
import re
import sys
from collections import defaultdict

import joblib
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import extract_house_number, extract_street_name, generate_features, normalize_address


class AddressMatcher:
    def __init__(self, addresses_df, use_index=True):
        self.df = addresses_df
        self.addresses = addresses_df['Адрес'].tolist()
        self.unoms = addresses_df['УНОМ'].tolist()
        self.ids = addresses_df['ID'].tolist()
        self.use_index = use_index

        print("Предварительная обработка адресов...")
        self.normalized_addresses = []
        for addr in self.addresses:
            try:
                norm_addr = normalize_address(addr)
                self.normalized_addresses.append(norm_addr)
            except Exception as e:
                print(f"Ошибка нормализации адреса: {addr}, ошибка: {e}")
                self.normalized_addresses.append("")

        if use_index:
            print("Создание хэш-индекса для быстрого поиска...")
            self.index = defaultdict(list)
            for idx, norm_addr in enumerate(self.normalized_addresses):
                if norm_addr:
                    self.index[norm_addr].append(idx)
                    street_only = ' '.join(norm_addr.split()[:-1]) if len(norm_addr.split()) > 1 else norm_addr
                    if street_only != norm_addr:
                        self.index[street_only].append(idx)

            print(f"Индекс создан: {len(self.index)} уникальных ключей")

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

    def extract_house_main_number(self, house_number):
        """Извлекает основной номер дома (без корпуса)"""
        if not house_number:
            return None
        match = re.match(r'(\d+)', str(house_number))
        return match.group(1) if match else None

    def extract_street_prefix(self, street_name, length=5):
        """Извлекает префикс улицы для быстрого сравнения"""
        if not street_name:
            return ""
        clean = re.sub(r'[^\w]', '', street_name.lower())
        return clean[:length]

    def exact_match_search(self, query_normalized):
        if not self.use_index:
            return None

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

    def fuzzy_search(self, query, query_normalized, query_house=None, top_n=20):
        results = process.extract(
            query_normalized,
            self.normalized_addresses,
            scorer=fuzz.ratio,
            limit=top_n,
            score_cutoff=50
        )

        # === ПРИОРИТЕТ ДЛЯ ЯВНОГО ТИПА УЛИЦЫ ===
        # Проверяем ОРИГИНАЛЬНЫЙ запрос (не нормализованный)
        query_original_lower = query.lower().strip()
        if query_original_lower.startswith(('ул ', 'улица ')):
            # Извлекаем название улицы из оригинального запроса
            match = re.search(r'(?:ул|улица)\s+([а-яА-ЯёЁ]+)', query_original_lower)
            if match:
                street_name = match.group(1)  # "таллинская"

                # Ищем адреса, где название улицы начинается с этого слова
                exact_street_matches = []
                for idx, addr_norm in enumerate(self.normalized_addresses):
                    # Получаем оригинальный адрес и извлекаем название улицы
                    addr_street = extract_street_name(self.addresses[idx]).lower()
                    # Сравниваем начало
                    if addr_street.startswith(street_name):
                        exact_street_matches.append((addr_norm, 95, idx))

                # Если нашли точные совпадения по названию улицы
                if exact_street_matches:
                    # Фильтруем результаты, оставляя только те, где название улицы совпадает
                    filtered_results = []
                    for addr_norm, score, idx in results:
                        addr_street = extract_street_name(self.addresses[idx]).lower()
                        if addr_street.startswith(street_name):
                            filtered_results.append((addr_norm, score, idx))

                    if filtered_results:
                        results = filtered_results
                    else:
                        results = exact_street_matches

        # Если ничего не найдено, пробуем поиск по ключевым словам
        if not results:
            words = query_normalized.split()
            street_words = [w for w in words if w not in ['дом', 'корпус', 'улица', 'бульвар'] and not w.isdigit()]
            numbers = [w for w in words if w.isdigit() or 'к' in w]

            if street_words and numbers:
                street_query = ' '.join(street_words)
                street_results = process.extract(
                    street_query,
                    self.normalized_addresses,
                    scorer=fuzz.ratio,
                    limit=top_n,
                    score_cutoff=60
                )

                for addr_norm, score, idx in street_results:
                    addr_house = extract_house_number(self.addresses[idx])
                    if numbers[0] in addr_house or addr_house in numbers[0]:
                        results.append((addr_norm, score, idx))

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

    def calculate_hybrid_score(self, query, candidate, features, has_explicit_type=False):
        """
        Гибридный расчет оценки с учетом контекста
        """
        query_street = extract_street_name(query)
        candidate_street = extract_street_name(candidate['address'])
        query_house = extract_house_number(query)
        candidate_house = extract_house_number(candidate['address'])

        # Базовые веса признаков
        weights = [0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.20, 0.05, 0.05, 0.10]

        # === 1. Быстрая фильтрация по номеру дома ===
        if query_house and candidate_house:
            query_main = self.extract_house_main_number(query_house)
            cand_main = self.extract_house_main_number(candidate_house)
            if query_main and cand_main and query_main != cand_main:
                return 0.0  # Разные основные номера - исключаем кандидата

        # === 2. Создаем копию признаков для модификации ===
        modified_features = features.copy()

        # === 3. Приоритет точного совпадения префикса улицы ===
        query_prefix = self.extract_street_prefix(query_street, 5)
        cand_prefix = self.extract_street_prefix(candidate_street, 5)

        if query_prefix and cand_prefix:
            if query_prefix == cand_prefix:
                # Префиксы совпадают - большой бонус
                modified_features[1] = min(1.0, modified_features[1] + 0.3)
                modified_features[4] = min(1.0, modified_features[4] + 0.2)
            elif query_prefix[0] != cand_prefix[0]:
                # Первая буква разная - сильный штраф
                modified_features[1] = modified_features[1] * 0.3
                modified_features[4] = modified_features[4] * 0.3

        # === 4. Если в запросе явно указан тип улицы ===
        if has_explicit_type:
            # Дополнительный бонус за точное совпадение
            if query_prefix == cand_prefix:
                modified_features[1] = min(1.0, modified_features[1] + 0.2)

        # === 5. Бонус за точное совпадение дома ===
        if query_house and candidate_house:
            if query_house == candidate_house:
                modified_features[6] = min(1.0, modified_features[6] + 0.3)
            else:
                query_main = self.extract_house_main_number(query_house)
                cand_main = self.extract_house_main_number(candidate_house)
                if query_main and cand_main and query_main == cand_main:
                    modified_features[6] = min(1.0, modified_features[6] + 0.15)

        # === 6. Вычисляем взвешенную сумму ===
        score = sum(f * w for f, w in zip(modified_features, weights))

        return min(1.0, score)

    def find_best_match(self, query, top_n=20):
        query_normalized = normalize_address(query)

        # Определяем, есть ли явный тип улицы в начале (по оригинальному запросу)
        has_explicit_type = bool(re.search(r'^(ул|улица|проспект|бульвар|переулок|пр-т|б-р)', query.lower().strip()))

        exact_matches = self.exact_match_search(query_normalized)

        if exact_matches and len(exact_matches) == 1:
            best = exact_matches[0]
            best['final_score'] = 1.0
            best['ml_score'] = 1.0
            return [best]
        elif exact_matches and len(exact_matches) > 1:
            candidates = exact_matches
        else:
            # Передаем оригинальный query и нормализованный
            candidates = self.fuzzy_search(query, query_normalized, top_n)

        if not candidates:
            return []

        if self.model_loaded and len(candidates) > 0:
            features_matrix = []
            for candidate in candidates:
                features = generate_features(query, candidate['address'])
                features_matrix.append(features)

            features_matrix_scaled = self.scaler.transform(features_matrix)
            probabilities = self.model.predict_proba(features_matrix_scaled)[:, 1]

            for i, candidate in enumerate(candidates):
                candidate['ml_score'] = probabilities[i]

                features = generate_features(query, candidate['address'])
                hybrid_score = self.calculate_hybrid_score(query, candidate, features, has_explicit_type)

                if candidate.get('exact_match', False):
                    candidate['final_score'] = 0.8 + 0.2 * hybrid_score
                else:
                    # Комбинируем fuzzy, ML и hybrid score
                    candidate['final_score'] = (
                        0.25 * (candidate['fuzzy_score'] / 100) +
                        0.35 * candidate['ml_score'] +
                        0.40 * hybrid_score
                    )

            candidates.sort(key=lambda x: x['final_score'], reverse=True)
        else:
            for candidate in candidates:
                candidate['final_score'] = candidate['fuzzy_score'] / 100
                candidate['ml_score'] = candidate['fuzzy_score'] / 100
            candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return candidates

    def debug_fuzzy_search(self, query):
        """Отладочная версия fuzzy_search для диагностики проблем"""
        query_normalized = normalize_address(query)
        query_original_lower = query.lower().strip()

        print("\n=== ОТЛАДКА fuzzy_search ===")
        print(f"Оригинальный запрос: {query}")
        print(f"Нормализованный: {query_normalized}")
        print(f"Начинается с 'ул '? {query_original_lower.startswith(('ул ', 'улица '))}")

        if query_original_lower.startswith(('ул ', 'улица ')):
            match = re.search(r'(?:ул|улица)\s+([а-яА-ЯёЁ]+)', query_original_lower)
            if match:
                print(f"Извлеченное название улицы: {match.group(1)}")

        results = process.extract(
            query_normalized,
            self.normalized_addresses,
            scorer=fuzz.ratio,
            limit=10,
            score_cutoff=50
        )

        print("\nТоп-5 результатов fuzzy поиска:")
        for addr_norm, score, idx in results[:5]:
            addr_street = extract_street_name(self.addresses[idx]).lower()
            print(f"  {score:.1f}% - {self.addresses[idx][:50]}... (улица: {addr_street})")

        return results

    def search(self, query):
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
        results = []
        for query in queries:
            candidates = self.find_best_match(query)
            if candidates:
                results.append({
                    'query': query,
                    'unom': candidates[0]['unom'],
                    'address': candidates[0]['address'],
                    'score': candidates[0]['final_score']
                })
        return results

def main():
    print("Загрузка базы адресов...")
    df = pd.read_csv('data/addresses.csv')
    print(f"Загружено {len(df)} адресов")

    required_columns = ['Адрес', 'УНОМ']
    for col in required_columns:
        if col not in df.columns:
            print(f"Ошибка: В файле отсутствует колонка '{col}'")
            print(f"Доступные колонки: {list(df.columns)}")
            return

    matcher = AddressMatcher(df, use_index=True)

    print("\n" + "="*50)
    print("Address Matcher готов к работе")
    print("Введите 'exit' для выхода, 'batch' для пакетного режима")
    print("'debug' для отладки fuzzy_search")
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
        elif query.lower() == 'debug':
            debug_query = input("Введите адрес для отладки: ").strip()
            if debug_query:
                matcher.debug_fuzzy_search(debug_query)
        elif query:
            matcher.search(query)

if __name__ == "__main__":
    main()
