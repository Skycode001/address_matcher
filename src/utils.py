import re

import pandas as pd
from rapidfuzz import fuzz

# Словарь сокращений (расширенный)
ABBREVIATIONS = {
    'ул': 'улица',
    'улица': 'улица',
    'пр-т': 'проспект',
    'пр': 'проспект',
    'проспект': 'проспект',
    'пер': 'переулок',
    'переулок': 'переулок',
    'пл': 'площадь',
    'площадь': 'площадь',
    'б-р': 'бульвар',
    'бул': 'бульвар',
    'бульв': 'бульвар',
    'бульвар': 'бульвар',
    'наб': 'набережная',
    'набережная': 'набережная',
    'ш': 'шоссе',
    'шоссе': 'шоссе',
    'д': 'дом',
    'дом': 'дом',
    'к': 'корпус',
    'кп': 'корпус',
    'корп': 'корпус',
    'корпуса': 'корпус',
    'корпус': 'корпус',
    'кор': 'корпус',        # <--- НОВОЕ
    'кор.': 'корпус',       # <--- НОВОЕ
    'стр': 'строение',
    'строение': 'строение',
    'владение': 'владение',
    'вл': 'владение',
    'лит': 'литера',
    'литера': 'литера',
}

# Фонетическая карта
PHONETIC_MAP = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'i', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
}

def phonetic_normalize(text):
    if not text:
        return ""
    text = text.lower()
    result = []
    for char in text:
        if char in PHONETIC_MAP:
            result.append(PHONETIC_MAP[char])
        elif char.isdigit():
            result.append(char)
        else:
            result.append(char)
    return ''.join(result)[:10]

def normalize_address(address):
    """
    Универсальная нормализация адреса
    """
    if pd.isna(address):
        return ""

    address = str(address).lower().strip()

    # 1. Удаляем запятые и лишние пробелы
    address = address.replace(',', ' ')
    address = re.sub(r'\s+', ' ', address)

    # 2. Словарь типов улиц
    street_types = {
        'бульвар': ['бульвар', 'бул', 'бульв', 'б-р'],
        'улица': ['улица', 'ул'],
        'проспект': ['проспект', 'пр', 'пр-т'],
        'переулок': ['переулок', 'пер'],
        'площадь': ['площадь', 'пл'],
        'набережная': ['набережная', 'наб'],
        'шоссе': ['шоссе', 'ш'],
        'проезд': ['проезд', 'пр-д'],
        'тупик': ['тупик', 'туп'],
    }

    # 3. Перестановка типа улицы из начала в конец
    words = address.split()
    if words:
        first_word = words[0].rstrip('.')
        for main_type, variants in street_types.items():
            if first_word in variants:
                remaining = ' '.join(words[1:])
                address = f"{remaining} {main_type}"
                break

    # 4. НОВАЯ ОБРАБОТКА: все варианты корпусов ДО удаления точек
    # кор. 1 -> корпус 1
    address = re.sub(r'кор\.\s+(\d+)', r'корпус \1', address)
    # кор.1 -> корпус 1
    address = re.sub(r'кор\.(\d+)', r'корпус \1', address)
    # кор 1 (без точки) -> корпус 1  <--- НОВЫЙ ПАТТЕРН
    address = re.sub(r'кор\s+(\d+)', r'корпус \1', address)
    # кор1 (без пробела) -> корпус 1
    address = re.sub(r'кор(\d+)', r'корпус \1', address)
    # корп 1 -> корпус 1
    address = re.sub(r'корп\s+(\d+)', r'корпус \1', address)
    # корп1 -> корпус 1
    address = re.sub(r'корп(\d+)', r'корпус \1', address)

    # 5. Удаляем остальные точки
    address = re.sub(r'\.', '', address)

    # 6. Обработка корпусов (после удаления точек)
    address = re.sub(r'(\d+)[кк](\d+)', r'\1 корпус \2', address)
    address = re.sub(r'(\d+)/(\d+)', r'\1 корпус \2', address)
    address = re.sub(r'(\d+)-(\d+)', r'\1 корпус \2', address)

    # 7. Дополнительная обработка для паттернов с пробелами
    address = re.sub(r'(\s+)(\d+)\s+к(\d+)', r'\1дом \2 корпус \3', address)
    address = re.sub(r'(\s+)(\d+)\s+к\s+(\d+)', r'\1дом \2 корпус \3', address)
    address = re.sub(r'^(\d+)\s+к(\d+)', r'дом \1 корпус \2', address)
    address = re.sub(r'^(\d+)\s+к\s+(\d+)', r'дом \1 корпус \2', address)

    # 8. Добавляем "дом" перед числами, если его нет
    if not re.search(r'(дом|д)\s*\d+', address):
        address = re.sub(r'(\s+)(\d+)(\s+корпус|\s+$|$)', r'\1дом \2\3', address)
        address = re.sub(r'^(\d+)(\s+корпус|\s+$|$)', r'дом \1\2', address)
        address = re.sub(r'(\d+)(корпус)', r'дом \1 \2', address)

    # 9. Нормализуем "д" в "дом"
    address = re.sub(r'д\s+(\d+)', r'дом \1', address)

    # 10. Удаляем лишние символы
    address = re.sub(r'[^\w\s]', ' ', address)
    address = re.sub(r'\s+', ' ', address)

    # 11. Заменяем сокращения
    words = address.split()
    normalized_words = []

    for word in words:
        replaced = False
        for main_type, variants in street_types.items():
            if word in variants:
                normalized_words.append(main_type)
                replaced = True
                break

        if not replaced:
            if word in ABBREVIATIONS:
                normalized_words.append(ABBREVIATIONS[word])
            else:
                normalized_words.append(word)

    result = ' '.join(normalized_words)
    result = re.sub(r'\s+', ' ', result)

    return result

def extract_house_number(address):
    """Извлечение номера дома с корпусом"""
    address = normalize_address(address)

    patterns = [
        r'дом\s+(\d+(?:[а-я])?)\s+корпус\s+(\d+(?:[а-я])?)',
        r'дом\s+(\d+(?:[а-я])?)к(\d+(?:[а-я])?)',
        r'д\.?\s*(\d+(?:[а-я])?)\s*к\.?\s*(\d+(?:[а-я])?)',
        r'(\d+(?:[а-я])?)\s+корпус\s+(\d+(?:[а-я])?)',
        r'(\d+(?:[а-я])?)к(\d+(?:[а-я])?)',
        r'(\d+(?:[а-я])?)/(\d+(?:[а-я])?)',
        r'(\d+(?:[а-я])?)-(\d+(?:[а-я])?)',
        r'\b(\d+(?:[а-я])?)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, address)
        if match:
            groups = match.groups()
            if len(groups) >= 2 and groups[1] and groups[1].strip():
                return f"{groups[0]}к{groups[1]}"
            elif groups[0]:
                return groups[0]

    return ""

def extract_street_name(address):
    """Извлекает название улицы с приоритетом точного совпадения"""
    address = normalize_address(address)

    # Удаляем номера домов и корпусов
    address = re.sub(r'\s*дом\s*\d+(?:[а-я])?(?:\s*корпус\s*\d+(?:[а-я])?)?', '', address)
    address = re.sub(r'\s*д\.?\s*\d+(?:[а-я])?(?:\s*к\.?\s*\d+(?:[а-я])?)?', '', address)
    address = re.sub(r'\s*\d+(?:[а-я])?к\d+(?:[а-я])?', '', address)
    address = re.sub(r'\s*\d+(?:[а-я])?$', '', address)

    # Удаляем типы улиц для чистого названия
    for word in ['улица', 'проспект', 'переулок', 'площадь', 'бульвар', 'набережная', 'шоссе', 'проезд', 'тупик']:
        address = address.replace(word, '')

    street_name = address.strip()

    # Если осталось слово из 3+ букв - возвращаем
    if len(street_name) >= 3:
        return street_name

    # Иначе возвращаем исходную нормализованную строку
    return normalize_address(address).split()[0] if normalize_address(address).split() else ""

def generate_features(query, candidate):
    """Генерирует 10 признаков"""
    features = []

    query_norm = normalize_address(query)
    candidate_norm = normalize_address(candidate)
    query_street = extract_street_name(query)
    candidate_street = extract_street_name(candidate)
    query_house = extract_house_number(query)
    candidate_house = extract_house_number(candidate)

    features.append(fuzz.ratio(query_norm, candidate_norm) / 100)
    features.append(fuzz.ratio(query_street, candidate_street) / 100)
    features.append(fuzz.token_sort_ratio(query_street, candidate_street) / 100)
    features.append(fuzz.partial_ratio(query_street, candidate_street) / 100)

    query_phonetic = phonetic_normalize(query_street)
    candidate_phonetic = phonetic_normalize(candidate_street)
    features.append(fuzz.ratio(query_phonetic, candidate_phonetic) / 100)

    if len(query_street) >= 5 and len(candidate_street) >= 5:
        features.append(1 if query_street[:5] == candidate_street[:5] else 0)
    else:
        features.append(fuzz.ratio(query_street[:3] if len(query_street) >= 3 else query_street,
                                   candidate_street[:3] if len(candidate_street) >= 3 else candidate_street) / 100)

    if query_house and candidate_house:
        if query_house == candidate_house:
            features.append(1.0)
        else:
            query_main = re.match(r'(\d+)', query_house)
            cand_main = re.match(r'(\d+)', candidate_house)
            if query_main and cand_main and query_main.group(1) == cand_main.group(1):
                features.append(0.7)
            else:
                features.append(fuzz.ratio(query_house, candidate_house) / 100)
    else:
        features.append(0.0)

    query_has_corpus = 1 if 'к' in query_house else 0
    cand_has_corpus = 1 if 'к' in candidate_house else 0
    features.append(1 if query_has_corpus == cand_has_corpus else 0)

    features.append(1 - abs(len(query) - len(candidate)) / max(len(query), len(candidate), 1))

    query_type = 1 if any(x in query_norm for x in ['улица', 'проспект', 'переулок', 'площадь', 'бульвар', 'набережная', 'шоссе', 'проезд', 'тупик']) else 0
    cand_type = 1 if any(x in candidate_norm for x in ['улица', 'проспект', 'переулок', 'площадь', 'бульвар', 'набережная', 'шоссе', 'проезд', 'тупик']) else 0
    features.append(1 if query_type == cand_type else 0)

    return features
