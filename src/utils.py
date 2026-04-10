import re
import pandas as pd
from rapidfuzz import fuzz, process

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
    'стр': 'строение',
    'строение': 'строение',
    'владение': 'владение',
    'вл': 'владение',
    'лит': 'литера',
    'литера': 'литера',
}

# Фонетическая карта для русских букв
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
    Приводит адрес к нормализованному виду с учетом корпусов
    """
    if pd.isna(address):
        return ""
    
    address = str(address).lower().strip()
    
    # Обработка слитных слов "улицаТаллинская" -> "улица Таллинская"
    address = re.sub(r'(улица|ул|проспект|пр|переулок|пер|площадь|пл)([а-я]+)', r'\1 \2', address)
    
    # Обработка "ул Таллинская" -> "Таллинская улица" (меняем порядок)
    # Находим "ул" в начале и переносим в конец
    address = re.sub(r'^ул\s+', '', address)
    address = re.sub(r'\s+ул\s+', ' ', address)
    
    # Специальная обработка для корпусов
    address = re.sub(r'(\d+)[кк](\d+)', r'\1 корпус \2', address)
    address = re.sub(r'(\d+)/(\d+)', r'\1 корпус \2', address)
    address = re.sub(r'(\d+)-(\d+)', r'\1 корпус \2', address)
    address = re.sub(r'корп\s+(\d+)', r'корпус \1', address)
    address = re.sub(r'к\s+(\d+)', r'корпус \1', address)
    
    # Обработка домов
    address = re.sub(r'д\.?\s*(\d+)', r'дом \1', address)
    
    # Удаляем лишние символы
    address = re.sub(r'[^\w\s]', ' ', address)
    address = re.sub(r'\s+', ' ', address)
    
    # Заменяем сокращения
    words = address.split()
    normalized_words = []
    
    for word in words:
        if word in ABBREVIATIONS:
            normalized_words.append(ABBREVIATIONS[word])
        elif word.rstrip('.') in ABBREVIATIONS:
            normalized_words.append(ABBREVIATIONS[word.rstrip('.')])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)

def extract_house_number(address):
    """
    Извлечение номера дома с корпусом
    """
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
            if len(match.groups()) > 1 and match.group(2):
                return f"{match.group(1)}к{match.group(2)}"
            return match.group(1)
    
    return ""

def extract_street_name(address):
    """
    Извлекает название улицы
    """
    address = normalize_address(address)
    
    # Удаляем номера домов и служебные слова
    address = re.sub(r'\s*дом\s*\d+(?:[а-я])?(?:\s*корпус\s*\d+(?:[а-я])?)?', '', address)
    address = re.sub(r'\s*д\.?\s*\d+(?:[а-я])?(?:\s*к\.?\s*\d+(?:[а-я])?)?', '', address)
    address = re.sub(r'\s*\d+(?:[а-я])?к\d+(?:[а-я])?', '', address)
    address = re.sub(r'\s*\d+(?:[а-я])?$', '', address)
    
    # Удаляем служебные слова
    for word in ['улица', 'проспект', 'переулок', 'площадь', 'бульвар', 'набережная', 'шоссе']:
        address = address.replace(word, '')
    
    return address.strip()

def generate_features(query, candidate):
    """Генерирует РОВНО 10 признаков"""
    features = []
    
    query_norm = normalize_address(query)
    candidate_norm = normalize_address(candidate)
    query_street = extract_street_name(query)
    candidate_street = extract_street_name(candidate)
    query_house = extract_house_number(query)
    candidate_house = extract_house_number(candidate)
    
    # 1. Базовое совпадение
    features.append(fuzz.ratio(query_norm, candidate_norm) / 100)
    
    # 2. Совпадение улиц (простое)
    features.append(fuzz.ratio(query_street, candidate_street) / 100)
    
    # 3. Token sort ratio
    features.append(fuzz.token_sort_ratio(query_street, candidate_street) / 100)
    
    # 4. Partial ratio
    features.append(fuzz.partial_ratio(query_street, candidate_street) / 100)
    
    # 5. Фонетика
    query_phonetic = phonetic_normalize(query_street)
    candidate_phonetic = phonetic_normalize(candidate_street)
    features.append(fuzz.ratio(query_phonetic, candidate_phonetic) / 100)
    
    # 6. Первые 5 букв
    if len(query_street) >= 5 and len(candidate_street) >= 5:
        features.append(1 if query_street[:5] == candidate_street[:5] else 0)
    else:
        features.append(fuzz.ratio(query_street[:3] if len(query_street) >= 3 else query_street,
                                   candidate_street[:3] if len(candidate_street) >= 3 else candidate_street) / 100)
    
    # 7. Номер дома
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
    
    # 8. Наличие корпуса
    query_has_corpus = 1 if 'к' in query_house else 0
    cand_has_corpus = 1 if 'к' in candidate_house else 0
    features.append(1 if query_has_corpus == cand_has_corpus else 0)
    
    # 9. Разница в длине
    features.append(1 - abs(len(query) - len(candidate)) / max(len(query), len(candidate), 1))
    
    # 10. Совпадение типа улицы
    query_type = 1 if any(x in query_norm for x in ['улица', 'проспект', 'переулок']) else 0
    cand_type = 1 if any(x in candidate_norm for x in ['улица', 'проспект', 'переулок']) else 0
    features.append(1 if query_type == cand_type else 0)
    
    return features