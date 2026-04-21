"""
Модуль для определения и ранжирования адресов в ТиНАО (Троицкий и Новомосковский административные округа)
Логика: чем больше совпадений со справочниками, тем выше балл
"""

import re
from typing import Dict, List, Tuple

import pandas as pd

# Справочник "Город"
CITY_REF = {
    'Троицк': 'Троицк',
}

# Справочник "Внутригородская_территория"
INNER_CITY_REF = {
    'внутригородская территория городской округ Троицк': 'Троицк',
    'внутригородская территория поселение Первомайское': 'Первомайское',
    'внутригородская территория поселение Краснопахорское': 'Краснопахорское',
    'внутригородская территория поселение Киевский': 'Киевский',
    'внутригородская территория поселение Новофедоровское': 'Новофедоровское',
    'внутригородская территория поселение Михайлово-Ярцевское': 'Михайлово-Ярцевское',
    'внутригородская территория поселение Кленовское': 'Кленовское',
    'внутригородская территория поселение Роговское': 'Роговское',
    'внутригородская территория поселение Щаповское': 'Щаповское',
    'внутригородская территория поселение Вороновское': 'Вороновское',
    'внутригородская территория муниципальный округ Бекасово': 'Бекасово',
    'внутригородская территория муниципальный округ Краснопахорский': 'Краснопахорский',
    'внутригородская территория муниципальный округ Вороново': 'Вороново',
}

# Справочник "Муниципальное_образование"
MUNICIPALITY_REF = {
    'городской округ Троицк': 'Троицк',
    'поселение Первомайское': 'Первомайское',
    'поселение Щаповское': 'Щаповское',
    'поселение Краснопахорское': 'Краснопахорское',
    'поселение Вороновское': 'Вороновское',
    'поселение Кленовское': 'Кленовское',
    'поселение Михайлово-Ярцевское': 'Михайлово-Ярцевское',
    'поселение Роговское': 'Роговское',
    'поселение Новофедоровское': 'Новофедоровское',
    'поселение Киевский': 'Киевский',
    'муниципальный округ Бекасово': 'Бекасово',
    'муниципальный округ Краснопахорский': 'Краснопахорский',
    'муниципальный округ Вороново': 'Вороново',
}

# Справочник "Населенный_пункт"
SETTLEMENT_REF = {
    'деревня Раево': 'Раево',
    'деревня Романцево': 'Романцево',
    'деревня Пучково': 'Пучково',
    'деревня Фоминское': 'Фоминское',
    'деревня Конаково': 'Конаково',
    'деревня Дешино': 'Дешино',
    'деревня Троицкое': 'Троицкое',
    'деревня Семенково': 'Семенково',
    'деревня Белоусово': 'Белоусово',
    'посёлок ЛМС': 'ЛМС',
    'деревня Заболотье': 'Заболотье',
    'деревня Софьино': 'Софьино',
    'деревня Васюнино': 'Васюнино',
    'деревня Косовка': 'Косовка',
    'деревня Красная Пахра': 'Красная Пахра',
    'деревня Шаганино': 'Шаганино',
    'деревня Дубовка': 'Дубовка',
    'деревня Ивановское': 'Ивановское',
    'деревня Чернецкое': 'Чернецкое',
    'село Былово': 'Былово',
    'деревня Варварино': 'Варварино',
    'деревня Горчаково': 'Горчаково',
    'деревня Лужки': 'Лужки',
    'село Кленово': 'Кленово',
    'село Ознобишино': 'Ознобишино',
    'посёлок Птичное': 'Птичное',
    'деревня Поповка': 'Поповка',
    'деревня Жуковка': 'Жуковка',
    'посёлок Щапово': 'Щапово',
    'деревня Шахово': 'Шахово',
    'деревня Львово': 'Львово',
    'хутор Хутора Гуляевы': 'Хутора Гуляевы',
    'рабочий посёлок Киевский': 'Киевский',
    'деревня Страдань': 'Страдань',
    'деревня Ясенки': 'Ясенки',
    'деревня Терехово': 'Терехово',
    'деревня Дровнино': 'Дровнино',
    'деревня Городок': 'Городок',
    'деревня Акулово': 'Акулово',
    'деревня Каменка': 'Каменка',
    'деревня Бакланово': 'Бакланово',
    'деревня Клоково': 'Клоково',
    'деревня Ботаково': 'Ботаково',
    'деревня Ожигово': 'Ожигово',
    'деревня Ворсино': 'Ворсино',
    'деревня Уварово': 'Уварово',
    'посёлок Курилово': 'Курилово',
    'деревня Ширяево': 'Ширяево',
    'деревня Рассудово': 'Рассудово',
    'деревня Новомихайловское': 'Новомихайловское',
    'деревня Сенькино-Секерино': 'Сенькино-Секерино',
    'деревня Лукошкино': 'Лукошкино',
    'село Вороново': 'Вороново',
    'посёлок Подсобного Хозяйства Минзаг': 'Подсобного Хозяйства Минзаг',
    'деревня Батыбино': 'Батыбино',
    'деревня Исаково': 'Исаково',
    'деревня Юрьево': 'Юрьево',
    'деревня Шеломово': 'Шеломово',
    'посёлок Рогово': 'Рогово',
    'село Сальково': 'Сальково',
    'посёлок Шишкин Лес': 'Шишкин Лес',
    'деревня Десна': 'Десна',
    'деревня Рождественно': 'Рождественно',
    'деревня Архангельское': 'Архангельское',
    'деревня Кузнецово': 'Кузнецово',
    'деревня Юдановка': 'Юдановка',
    'деревня Писково': 'Писково',
    'деревня Кресты': 'Кресты',
    'посёлок Рассудово': 'Рассудово',
    'село Красная Пахра': 'Красная Пахра',
    'деревня Яковлевское': 'Яковлевское',
    'деревня Троица': 'Троица',
    'посёлок Круги': 'Круги',
    'деревня Руднево': 'Руднево',
    'деревня Хмырово': 'Хмырово',
    'деревня Зверево': 'Зверево',
    'деревня Пахорка': 'Пахорка',
    'деревня Кувекино': 'Кувекино',
    'деревня Яковлево': 'Яковлево',
    'деревня Киселёвка': 'Киселёвка',
    'деревня Конюшково': 'Конюшково',
    'деревня Ватутинки': 'Ватутинки',
    'деревня Кленовка': 'Кленовка',
    'посёлок Зосимова Пустынь': 'Зосимова Пустынь',
    'деревня Овечкино': 'Овечкино',
    'деревня Подосинки': 'Подосинки',
    'село Красное': 'Красное',
    'деревня Чириково': 'Чириково',
    'деревня Сатино-Русское': 'Сатино-Русское',
    'деревня Костишово': 'Костишово',
    'деревня Русино': 'Русино',
    'деревня Кузенево': 'Кузенево',
    'деревня Песье': 'Песье',
    'деревня Ярцево': 'Ярцево',
    'деревня Александрово': 'Александрово',
    'посёлок Михайловского Лесничества': 'Михайловского Лесничества',
    'деревня Сатино-Татарское': 'Сатино-Татарское',
    'деревня Колотилово': 'Колотилово',
    'деревня Иваньково': 'Иваньково',
    'деревня Поляны': 'Поляны',
    'село Михайловское': 'Михайловское',
    'посёлок Дорожно-Ремонтного Пункта - 3': 'Дорожно-Ремонтного Пункта - 3',
    'посёлок Дома Отдыха "Плесково"': 'Дома Отдыха "Плесково"',
    'деревня Шарапово': 'Шарапово',
    'деревня Малыгино': 'Малыгино',
    'деревня Юрово': 'Юрово',
    'посёлок Дома Отдыха "Пахра"': 'Дома Отдыха "Пахра"',
    'посёлок Секерино': 'Секерино',
    'хутор Шалово': 'Шалово',
    'посёлок Красное': 'Красное',
    'посёлок Капустинка': 'Капустинка',
    'деревня Ильино': 'Ильино',
    'деревня Тетеринки': 'Тетеринки',
    'деревня Бунчиха': 'Бунчиха',
    'деревня Никоново': 'Никоново',
    'деревня Свитино': 'Свитино',
    'деревня Дмитровка': 'Дмитровка',
    'деревня Давыдово': 'Давыдово',
    'деревня Богородское': 'Богородское',
    'деревня Сахарово': 'Сахарово',
    'село Богоявление': 'Богоявление',
    'село Покровское': 'Покровское',
    'деревня Круча': 'Круча',
    'деревня Киселёво': 'Киселёво',
    'деревня Климовка': 'Климовка',
    'село Свитино': 'Свитино',
    'деревня Товарищево': 'Товарищево',
    'деревня Маврино': 'Маврино',
    'деревня Жохово': 'Жохово',
    'деревня Старогромово': 'Старогромово',
    'деревня Починки': 'Починки',
    'деревня Коротыгино': 'Коротыгино',
    'деревня Мешково': 'Мешково',
    'деревня Зыбино': 'Зыбино',
    'деревня Вяткино': 'Вяткино',
    'деревня Чегодаево': 'Чегодаево',
    'деревня Юрьевка': 'Юрьевка',
    'деревня Новогромово': 'Новогромово',
    'деревня Голохвастово': 'Голохвастово',
    'деревня Рыжово': 'Рыжово',
    'деревня Бабенки': 'Бабенки',
    'деревня Спас-Купля': 'Спас-Купля',
    'деревня Кузовлево': 'Кузовлево',
    'деревня Петрово': 'Петрово',
    'деревня Горнево': 'Горнево',
    'деревня Лыковка': 'Лыковка',
    'деревня Лопатино': 'Лопатино',
    'посёлок Дома Отдыха "Вороново"': 'Дома Отдыха "Вороново"',
    'село Никольское': 'Никольское',
    'посёлок Спортбазы': 'Спортбазы',
    'деревня Пудово-Сипягино': 'Пудово-Сипягино',
    'деревня Филино': 'Филино',
    'деревня Безобразово': 'Безобразово',
    'деревня Власьево': 'Власьево',
    'посёлок Станции Мачихино': 'Станции Мачихино',
    'деревня Алымовка': 'Алымовка',
    'деревня Тупиково': 'Тупиково',
    'деревня Пенино': 'Пенино',
    'деревня Новинки': 'Новинки',
    'деревня Евсеево': 'Евсеево',
    'деревня Черепово': 'Черепово',
    'деревня Станиславль': 'Станиславль',
    'деревня Пыхчево': 'Пыхчево',
    'деревня Новиково': 'Новиково',
    'деревня Фёдоровское': 'Фёдоровское',
    'посёлок Разъезда Пожитково': 'Разъезда Пожитково',
    'деревня Игнатово': 'Игнатово',
    'хутор Талызина': 'Талызина',
    'деревня Малеевка': 'Малеевка',
    'деревня Лукино': 'Лукино',
    'посёлок Армейский': 'Армейский',
}

# Ключевые слова для определения ТиНАО (дают +5 баллов)
TINAO_KEYWORDS = ['тинао', 'троицк', 'новомосковский', 'троицкий']

# Объединяем все справочники для поиска
ALL_REFERENCES = {
    **CITY_REF,
    **INNER_CITY_REF,
    **MUNICIPALITY_REF,
    **SETTLEMENT_REF,
}


def calculate_tinao_score(address: str) -> Tuple[int, List[str]]:
    """
    Рассчитывает количество совпадений адреса со справочниками ТиНАО.
    
    Args:
        address: Адрес для проверки
        
    Returns:
        Tuple[int, List[str]]: (количество совпадений, список найденных соответствий)
    """
    if not address:
        return 0, []

    address_lower = address.lower()
    matches = []
    score = 0

    # Проверка по ключевым словам (большой вес +5)
    for keyword in TINAO_KEYWORDS:
        if keyword in address_lower:
            score += 5
            matches.append(f"ключевое слово '{keyword}'")

    # Проверка по справочникам (каждое совпадение +1)
    for ref_name, ref_value in ALL_REFERENCES.items():
        if ref_name.lower() in address_lower:
            score += 1
            matches.append(ref_name)

    return score, matches


def find_tinao_candidates_by_references(query: str, df: pd.DataFrame, top_n: int = 100) -> List[Dict]:
    """
    Ищет кандидатов в базе по совпадениям со справочниками ТиНАО.
    """
    if not query or df is None:
        return []

    query_lower = query.lower()
    candidates = []

    # Удаляем ключевые слова ТиНАО для сравнения
    search_query = query_lower
    for keyword in TINAO_KEYWORDS:
        search_query = search_query.replace(keyword, '').strip()
    search_query = re.sub(r'[, ]+', ' ', search_query).strip()
    search_query = search_query.replace('"', '').replace("'", "")

    print(f"\n[DEBUG] Поиск по ТиНАО для запроса: {query}")
    print(f"[DEBUG] Поисковая строка: {search_query}")

    # Разбиваем на ключевые слова
    keywords = []
    for w in search_query.split():
        if len(w) >= 2 or w.isdigit():
            keywords.append(w)
    print(f"[DEBUG] Ключевые слова: {keywords}")

    # Функция для извлечения чисел, исключая порядковые числительные
    def extract_numbers_ignore_ordinal(text):
        """Извлекает числа, игнорируя порядковые числительные (1-й, 2-я, 3-е)"""
        # Сначала убираем порядковые числительные
        text_without_ordinal = re.sub(r'\b\d+[-][яйе]\b', '', text)
        # Извлекаем оставшиеся числа
        return set(re.findall(r'\b(\d+)\b', text_without_ordinal))

    # Извлекаем все числа из запроса (игнорируя порядковые числительные)
    all_numbers_in_query = extract_numbers_ignore_ordinal(search_query)
    print(f"[DEBUG] Все числа в запросе (без порядковых): {all_numbers_in_query}")

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
            print(f"[DEBUG] Найден контекст: {match[0]}={match[1]}")

    # Извлекаем простые числа (как отдельные слова, не в составе порядковых)
    simple_numbers = []
    for w in search_query.split():
        if re.match(r'^\d+$', w):
            simple_numbers.append(w)
    print(f"[DEBUG] Простые числа: {simple_numbers}")

    # Извлекаем числа с буквами (например, 5А, 1Б/Н)
    numbers_with_letters = []
    for w in search_query.split():
        if re.match(r'^\d+[А-Яа-я/]+$', w):
            numbers_with_letters.append(w)
    print(f"[DEBUG] Числа с буквами: {numbers_with_letters}")

    addresses = df['Адрес'].tolist()
    unoms = df['УНОМ'].tolist()

    for idx, address in enumerate(addresses):
        if not isinstance(address, str):
            continue
        if pd.isna(address):
            continue
        if not address.strip():
            continue

        address_lower = address.lower()

        # Проверяем ключевые слова
        all_keywords_found = True
        for kw in keywords:
            if kw not in address_lower:
                all_keywords_found = False
                break

        if not all_keywords_found:
            continue

        # Проверяем контекстные числа
        context_match = True
        for ctx in numbers_with_context:
            pattern = rf'{ctx["type"]}\s+{re.escape(ctx["number"])}'
            if not re.search(pattern, address_lower):
                context_match = False
                break

        if not context_match:
            continue

        # Проверяем простые числа
        numbers_match = True
        for num in simple_numbers:
            found = False
            address_words = re.split(r'[, ]+', address_lower)
            for aw in address_words:
                if aw == num:
                    found = True
                    break
                if aw.startswith(num) and (len(aw) == len(num) or not aw[len(num)].isdigit()):
                    found = True
                    break
            if not found:
                numbers_match = False
                break

        if not numbers_match:
            continue

        # Проверяем числа с буквами
        letters_match = True
        for num_let in numbers_with_letters:
            if num_let not in address_lower:
                letters_match = False
                break

        if not letters_match:
            continue

        # Проверка на лишние числа (игнорируя порядковые числительные в адресе)
        all_numbers_in_address = extract_numbers_ignore_ordinal(address_lower)
        extra_numbers = all_numbers_in_address - all_numbers_in_query
        if extra_numbers:
            print(f"[DEBUG] Лишние числа в адресе (игнорируя порядковые): {extra_numbers}")
            continue

        candidates.append({
            'index': idx,
            'address': address,
            'unom': unoms[idx],
            'address_lower': address_lower
        })

    if not candidates:
        print("[DEBUG] Кандидатов не найдено")
        return []

    # Считаем баллы
    scored_candidates = []
    for c in candidates:
        score = len(keywords) + len(simple_numbers) + len(numbers_with_context) + len(numbers_with_letters)

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

        # Бонус за ТиНАО
        if any(kw in query_lower for kw in TINAO_KEYWORDS):
            if 'троицк' in c['address_lower'] or 'городской округ троицк' in c['address_lower']:
                score += 10

        scored_candidates.append({
            'index': c['index'],
            'address': c['address'],
            'unom': c['unom'],
            'tinao_score': score
        })

    scored_candidates.sort(key=lambda x: x['tinao_score'], reverse=True)

    print("[DEBUG] Топ-5 кандидатов по баллам:")
    for i, c in enumerate(scored_candidates[:5]):
        print(f"  {i+1}. score={c['tinao_score']}: {c['address'][:80]}...")

    return scored_candidates[:top_n]


def rank_candidates_by_tinao(candidates: List[Dict], query: str) -> List[Dict]:
    """
    Ранжирует кандидатов по количеству совпадений со справочниками ТиНАО.
    
    Args:
        candidates: Список кандидатов (адресов из базы)
        query: Исходный запрос пользователя
        
    Returns:
        List[Dict]: Отсортированный список кандидатов
    """
    if not candidates:
        return candidates

    query_lower = query.lower()

    for candidate in candidates:
        address_lower = candidate['address'].lower()
        tinao_score = candidate.get('tinao_score', 0)

        # Проверяем ключевые слова ТиНАО
        for keyword in TINAO_KEYWORDS:
            if keyword in query_lower or keyword in address_lower:
                tinao_score += 5
                if 'tinao_matches' not in candidate:
                    candidate['tinao_matches'] = []
                candidate['tinao_matches'].append(keyword)

        # Проверяем совпадения со справочниками
        for ref_name, ref_value in ALL_REFERENCES.items():
            ref_lower = ref_name.lower()
            if ref_lower in query_lower or ref_lower in address_lower:
                tinao_score += 1
                if 'tinao_matches' not in candidate:
                    candidate['tinao_matches'] = []
                if ref_name not in candidate['tinao_matches']:
                    candidate['tinao_matches'].append(ref_name)

        candidate['tinao_score'] = tinao_score

    # Сортируем по tinao_score (по убыванию)
    candidates.sort(key=lambda x: x.get('tinao_score', 0), reverse=True)

    return candidates


def is_tinao_address(address: str, threshold: int = 2) -> bool:
    """
    Определяет, относится ли адрес к ТиНАО.
    
    Args:
        address: Адрес для проверки
        threshold: Минимальное количество совпадений для уверенного определения
        
    Returns:
        bool: True если адрес относится к ТиНАО
    """
    score, _ = calculate_tinao_score(address)
    return score >= threshold


def get_tinao_score_details(address: str) -> Dict:
    """
    Возвращает детальную информацию о принадлежности адреса к ТиНАО.
    
    Args:
        address: Адрес для проверки
        
    Returns:
        Dict: Словарь с полями:
            - is_tinao: bool
            - score: int
            - matches: List[str]
            - confidence: str
    """
    score, matches = calculate_tinao_score(address)

    if score >= 5:
        confidence = "высокая"
    elif score >= 2:
        confidence = "средняя"
    else:
        confidence = "низкая"

    return {
        'is_tinao': score >= 2,
        'score': score,
        'matches': matches,
        'confidence': confidence
    }


def normalize_tinao_query(query: str) -> str:
    """
    Нормализует запрос для поиска в ТиНАО.
    Убирает ключевые слова ТиНАО, оставляя только адресную часть.
    
    Пример:
    "троицк поселение Краснопахорское, деревня Раево, Ирландская улица, 1А"
    -> "поселение Краснопахорское, деревня Раево, Ирландская улица, 1А"
    """
    if not query:
        return query

    result = query.lower()

    # Убираем ключевые слова ТиНАО
    for keyword in TINAO_KEYWORDS:
        result = re.sub(r'\b' + re.escape(keyword) + r'\b', '', result)

    # Убираем лишние запятые и пробелы
    result = re.sub(r',\s*', ', ', result)
    result = re.sub(r'\s+', ' ', result)
    result = result.strip(' ,')

    return result
