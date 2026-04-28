"""
Модуль для определения и ранжирования адресов в Зеленограде (Зеленоградский административный округ)
Логика: ключевые слова 'зеленоград' и 'зелао' включают фильтрацию по округу
"""

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Добавляем путь для импорта из utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import normalize_address

# Справочник "Город"
CITY_REF = {
    'город Зеленоград': 'Зеленоград',
    'зеленоград': 'Зеленоград',
}

# Справочник "Внутригородская_территория"
INNER_CITY_REF = {
    'внутригородская территория муниципальный округ Силино': 'Силино',
    'внутригородская территория муниципальный округ Матушкино': 'Матушкино',
    'внутригородская территория муниципальный округ Савелки': 'Савелки',
    'внутригородская территория муниципальный округ Крюково': 'Крюково',
    'внутригородская территория муниципальный округ Старое Крюково': 'Старое Крюково',
}

# Справочник "Муниципальное_образование"
MUNICIPALITY_REF = {
    'муниципальный округ Силино': 'Силино',
    'муниципальный округ Матушкино': 'Матушкино',
    'муниципальный округ Савелки': 'Савелки',
    'муниципальный округ Крюково': 'Крюково',
    'муниципальный округ Старое Крюково': 'Старое Крюково',
}

# Справочник "Населенный_пункт"
SETTLEMENT_REF = {
    'посёлок Назарьево': 'Назарьево',
    'посёлок Рожки': 'Рожки',
    'деревня Кутузово': 'Кутузово',
    'посёлок Малино': 'Малино',
}

# Округа ЗелАО для фильтрации
ZELAO_DISTRICTS = [
    'Зеленоградский административный округ',
]

# Ключевые слова для определения ЗелАО (только для фильтрации, НЕ добавляют баллов)
ZELAO_KEYWORDS = ['зеленоград', 'зелао']

# Объединяем все справочники для поиска
ALL_REFERENCES = {
    **CITY_REF,
    **INNER_CITY_REF,
    **MUNICIPALITY_REF,
    **SETTLEMENT_REF,
}

# Синонимы для нормализации ЗелАО
ZELAO_SYNONYMS = {}


def normalize_zelao_text(text: str) -> str:
    """
    Нормализует текст для поиска в ЗелАО с использованием основной нормализации.
    """
    if not text:
        return ""
    result = normalize_address(text, apply_reverse=False)

    # Удаляем стоп-слова
    result = re.sub(r'\bрайон\b', '', result)
    result = re.sub(r'\bр-н\b', '', result)
    result = re.sub(r'\bрн\b', '', result)

    # Очищаем лишние пробелы
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def is_zelao_query(query: str) -> bool:
    """
    Проверяет, является ли запрос запросом по ЗелАО.
    Возвращает True, если в запросе есть ключевые слова 'зеленоград' или 'зелао'.
    """
    if not query:
        return False
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ZELAO_KEYWORDS)


def filter_by_zelao_district(df: pd.DataFrame) -> pd.DataFrame:
    """
    Фильтрует DataFrame по округу ЗелАО.
    """
    if df is None or df.empty:
        return df

    if 'Округ' not in df.columns:
        print("[WARNING] В базе отсутствует столбец 'Округ'. Фильтрация по ЗелАО невозможна.")
        return df

    mask = df['Округ'].isin(ZELAO_DISTRICTS)
    return df[mask].copy()


def calculate_zelao_score(address: str) -> Tuple[int, List[str]]:
    """
    Рассчитывает количество совпадений адреса со справочниками ЗелАО.
    """
    if not address:
        return 0, []

    address_lower = address.lower()
    matches = []
    score = 0

    for ref_name, ref_value in ALL_REFERENCES.items():
        if ref_name.lower() in address_lower:
            score += 1
            matches.append(ref_name)

    return score, matches


def find_zelao_candidates_by_references(
    query: str,
    df: pd.DataFrame,
    normalized_addresses: Optional[List[str]] = None,
    top_n: int = 100
) -> List[Dict]:
    """
    Ищет кандидатов в отфильтрованной по округу базе по совпадениям со справочниками ЗелАО.
    Args:
        query: Запрос пользователя
        df: DataFrame с адресами
        normalized_addresses: Предварительно нормализованные адреса (для ускорения)
        top_n: Количество кандидатов
    """
    if not query or df is None or df.empty:
        return []

    # Фильтруем базу по округу ЗелАО
    filtered_df = filter_by_zelao_district(df)

    if filtered_df.empty:
        return []

    query_lower = query.lower()

    # Нормализуем запрос целиком
    query_normalized = normalize_zelao_text(query_lower)

    # Удаляем ключевые слова ЗелАО для сравнения
    search_query = query_normalized
    for keyword in ZELAO_KEYWORDS:
        search_query = search_query.replace(keyword, '').strip()
    search_query = re.sub(r'[, ]+', ' ', search_query).strip()
    search_query = search_query.replace('"', '').replace("'", "")

    # Разбиваем на ключевые слова
    keywords = []
    for w in search_query.split():
        if len(w) >= 2 or w.isdigit() or w.replace('-', '').isdigit():
            keywords.append(w)

    # Извлекаем порядковые числительные (6-я, 5-й и т.д.)
    ordinal_pattern = re.compile(r'\b(\d+)-[яйе]\b')
    ordinals_in_query = ordinal_pattern.findall(search_query)

    # Функция для извлечения чисел, исключая порядковые числительные
    def extract_numbers_ignore_ordinal(text):
        text_without_ordinal = re.sub(r'\b\d+[-][яйе]\b', '', text)
        return set(re.findall(r'\b(\d+)\b', text_without_ordinal))

    all_numbers_in_query = extract_numbers_ignore_ordinal(search_query)

    # Извлекаем все числа и их контекст из запроса
    numbers_with_context = []
    patterns = [
        r'(строение|сооружение)\s+([0-9А-Яа-я/]+)',
        r'(владение)\s+(\d+)',
        r'(участок)\s+([0-9А-Яа-я/]+)',
        r'(дом|д)\s+(\d+[А-Яа-я]?)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, search_query)
        for match in matches:
            numbers_with_context.append({
                'type': match[0],
                'number': match[1]
            })

    # Извлекаем простые числа
    simple_numbers = [w for w in search_query.split() if re.match(r'^\d+$', w)]

    # Извлекаем числа с буквами
    numbers_with_letters = [w for w in search_query.split() if re.match(r'^\d+[А-Яа-я/]+$', w)]

    addresses = filtered_df['Адрес'].tolist()
    unoms = filtered_df['УНОМ'].tolist()

    # Используем предварительно нормализованные адреса или нормализуем сейчас
    if normalized_addresses is not None and len(normalized_addresses) == len(df):
        # Фильтруем нормализованные адреса по тем же индексам
        filtered_indices = filtered_df.index.tolist()
        filtered_normalized = [normalized_addresses[i] for i in filtered_indices]
    else:
        filtered_normalized = [normalize_zelao_text(addr) for addr in addresses]

    candidates = []

    for idx, (address, address_normalized) in enumerate(zip(addresses, filtered_normalized)):
        if not isinstance(address, str) or pd.isna(address) or not address.strip():
            continue

        address_lower = address.lower()

        # Проверяем ключевые слова (в нормализованном виде)
        # Требуем совпадения ВСЕХ ключевых слов
        all_keywords_found = True
        for kw in keywords:
            if kw in ordinals_in_query:
                continue

            # Прямая проверка
            if kw in address_normalized:
                continue

            # Специальная проверка: если это название (не служебное слово)
            # Проверяем в формате "муниципальный округ X"
            stop_words = {'деревня', 'поселок', 'село', 'город', 'дом', 'улица',
                         'проспект', 'бульвар', 'переулок', 'шоссе', 'набережная',
                         'площадь', 'микрорайон', 'строение', 'корпус', 'владение'}

            if kw not in stop_words and len(kw) > 2:
                # Ищем в формате "муниципальный округ X" или "городской округ X"
                if re.search(r'муниципальный\s+округ\s+' + re.escape(kw), address_normalized):
                    continue
                if re.search(r'городской\s+округ\s+' + re.escape(kw), address_normalized):
                    continue

            # Если ничего не подошло - слово не найдено
            all_keywords_found = False
            break

        if not all_keywords_found:
            continue

        # Проверяем порядковые числительные
        ordinals_match = True
        for ordinal in ordinals_in_query:
            if f"{ordinal}-" not in address_normalized and f"{ordinal} " not in address_normalized:
                ordinals_match = False
                break

        if not ordinals_match:
            continue

        # Проверяем контекстные числа
        context_match = True
        for ctx in numbers_with_context:
            pattern = rf'{ctx["type"]}\s+{re.escape(ctx["number"])}'
            if not re.search(pattern, address_normalized):
                context_match = False
                break

        if not context_match:
            continue

        # Проверяем простые числа
        numbers_match = True
        for num in simple_numbers:
            if num not in address_normalized:
                numbers_match = False
                break

        if not numbers_match:
            continue

        # Проверяем числа с буквами
        letters_match = True
        for num_let in numbers_with_letters:
            if num_let not in address_normalized:
                letters_match = False
                break

        if not letters_match:
            continue

        # Проверка на лишние числа
        all_numbers_in_address = extract_numbers_ignore_ordinal(address_normalized)
        extra_numbers = all_numbers_in_address - all_numbers_in_query
        if extra_numbers:
            continue

        # ===== ПРОВЕРКА НА ЛИШНИЕ БУКВЫ В НОМЕРЕ ДОМА =====
        # Если в запросе число без буквы, а в адресе число с буквой - пропускаем
        reject_due_to_letter = False
        if simple_numbers:
            for num in simple_numbers:
                # Ищем паттерн: число + буква (например, 13Б, 13А)
                pattern_with_letter = rf'\b{num}[а-яА-Я]\b'
                if re.search(pattern_with_letter, address_lower):
                    reject_due_to_letter = True
                    break
        if reject_due_to_letter:
            continue

        candidates.append({
            'index': filtered_df.index[idx],
            'address': address,
            'unom': unoms[idx],
            'address_lower': address_lower,
            'address_normalized': address_normalized
        })

    if not candidates:
        return []

    # ===== ПРОВЕРКА НА НЕОДНОЗНАЧНОСТЬ =====
    # Если в запросе нет уточняющего слова (деревня, посёлок, село, город и т.д.),
    # и есть несколько кандидатов с разными населенными пунктами - ничего не находим

    location_keywords = ['деревня', 'посёлок', 'поселок', 'село', 'город', 'городской', 'хутор']

    has_location_keyword = any(kw in query_lower for kw in location_keywords)

    if not has_location_keyword:
        # Проверяем, есть ли несколько разных населенных пунктов среди кандидатов
        settlements = set()

        for c in candidates:
            settlement = None
            for kw in location_keywords:
                # Ищем паттерн: "деревня Название"
                match = re.search(rf'{kw}\s+([а-яА-ЯёЁ\-]+)', c['address_lower'])
                if match:
                    settlement = match.group(1).lower()
                    break
            if settlement:
                settlements.add(settlement)

        if len(settlements) > 1:
            return []

    # Считаем баллы
    scored_candidates = []
    for c in candidates:
        score = len(keywords) + len(ordinals_in_query) + len(simple_numbers) + len(numbers_with_context) + len(numbers_with_letters)

        # Бонус за населенный пункт
        for settlement in SETTLEMENT_REF.keys():
            if settlement.lower() in query_lower and settlement.lower() in c['address_lower']:
                score += 20
                break

        # Бонус за муниципалитет
        for mun in MUNICIPALITY_REF.keys():
            if mun.lower() in query_lower and mun.lower() in c['address_lower']:
                score += 15
                break

        # Бонус за точное совпадение номера дома
        house_match = re.search(r'дом\s+(\d+)', c['address_lower'])
        if house_match:
            house_number = house_match.group(1)
            if house_number in all_numbers_in_query:
                score += 30

        # Бонус за совпадение микрорайона
        if 'микрорайон' in query_lower:
            micro_match = re.search(r'микрорайон\s+["\']?([а-яА-Я\d]+)["\']?', query_lower)
            if micro_match:
                micro_name = micro_match.group(1).lower()
                if micro_name in c['address_lower']:
                    score += 25

        scored_candidates.append({
            'index': c['index'],
            'address': c['address'],
            'unom': c['unom'],
            'zelao_score': score
        })

    scored_candidates.sort(key=lambda x: x['zelao_score'], reverse=True)

    return scored_candidates[:top_n]


def rank_candidates_by_zelao(candidates: List[Dict], query: str) -> List[Dict]:
    """
    Ранжирует кандидатов по количеству совпадений со справочниками ЗелАО.
    """
    if not candidates:
        return candidates

    query_lower = query.lower()

    for candidate in candidates:
        address_lower = candidate['address'].lower()
        zelao_score = candidate.get('zelao_score', 0)

        for ref_name, ref_value in ALL_REFERENCES.items():
            ref_lower = ref_name.lower()
            if ref_lower in query_lower or ref_lower in address_lower:
                zelao_score += 1
                if 'zelao_matches' not in candidate:
                    candidate['zelao_matches'] = []
                if ref_name not in candidate['zelao_matches']:
                    candidate['zelao_matches'].append(ref_name)

        candidate['zelao_score'] = zelao_score

    candidates.sort(key=lambda x: x.get('zelao_score', 0), reverse=True)
    return candidates


def is_zelao_address(address: str, threshold: int = 2) -> bool:
    """
    Определяет, относится ли адрес к ЗелАО по совпадениям со справочниками.
    """
    score, _ = calculate_zelao_score(address)
    return score >= threshold


def get_zelao_score_details(address: str) -> Dict:
    """
    Возвращает детальную информацию о принадлежности адреса к ЗелАО.
    """
    score, matches = calculate_zelao_score(address)

    if score >= 5:
        confidence = "высокая"
    elif score >= 2:
        confidence = "средняя"
    else:
        confidence = "низкая"

    return {
        'is_zelao': score >= 2,
        'score': score,
        'matches': matches,
        'confidence': confidence
    }


def normalize_zelao_query(query: str) -> str:
    """
    Нормализует запрос для поиска в ЗелАО.
    """
    if not query:
        return query

    result = query.lower()

    for keyword in ZELAO_KEYWORDS:
        result = re.sub(r'\b' + re.escape(keyword) + r'\b', '', result)

    result = re.sub(r',\s*', ', ', result)
    result = re.sub(r'\s+', ' ', result)
    result = result.strip(' ,')

    return result
