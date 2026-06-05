#!/usr/bin/env python3
"""
Address Matcher - система поиска адресов
"""

import argparse
import os
import sys
import warnings

import pandas as pd

# Добавляем путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.matcher import AddressMatcher

# Игнорируем предупреждение о смешанных типах в колонках при загрузке CSV
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


def print_banner():
    """Выводит приветственный баннер"""
    print("=" * 60)
    print("   Address Matcher v1.0")
    print("   Система поиска адресов по базе данных")
    print("=" * 60)
    print()
    print("Поддерживаемые округа:")
    print("   • Вся Москва (по умолчанию)")
    print("   • ТиНАО (Троицкий и Новомосковский) - добавьте 'тинао'")
    print("   • ЗелАО (Зеленоград) - добавьте 'зеленоград' или 'зелао'")
    print("=" * 60)


def print_menu():
    """Выводит меню выбора режима"""
    print("\n" + "-" * 40)
    print("Выберите режим работы:")
    print("   1. Интерактивный режим (ручной ввод адресов)")
    print("   2. Файловый режим (обогащение файла с адресами)")
    print("   3. Тестовый режим (запуск диагностики)")
    print("   0. Выход")
    print("-" * 40)


def interactive_mode(matcher):
    """
    Интерактивный режим - ручной ввод адресов
    """
    print("\n" + "=" * 50)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Введите 'exit' для выхода, 'menu' для возврата в главное меню")
    print("=" * 50)

    while True:
        query = input("\nВведите адрес: ").strip()
        if query.lower() == 'exit':
            print("До свидания!")
            return False
        elif query.lower() == 'menu':
            return True
        elif query:
            matcher.search(query)


def check_data_file(data_dir: str = "data") -> tuple:
    """
    Проверяет наличие файла данных в папке data.
    Args:
        data_dir: Путь к папке с данными
    Returns:
        tuple: (file_path, file_type) или (None, None) если файл не найден
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return None, None

    # Ищем файл с названием "Data" (любое расширение)
    possible_files = []
    for file in os.listdir(data_dir):
        name, ext = os.path.splitext(file)
        if name.lower() == 'data':
            possible_files.append((file, ext.lower()))

    if not possible_files:
        return None, None

    # Берем первый подходящий файл
    file_name, ext = possible_files[0]
    file_path = os.path.join(data_dir, file_name)

    return file_path, ext


def file_mode(matcher):
    """
    Файловый режим - обогащение файла с адресами данными из базы
    """
    print("\n" + "=" * 60)
    print("ФАЙЛОВЫЙ РЕЖИМ ОБОГАЩЕНИЯ ДАННЫХ")
    print("=" * 60)

    # Проверяем наличие файла
    file_path, file_ext = check_data_file()

    if file_path is None:
        print("\n❌ Файл не найден!")
        print("\nИнструкция:")
        print("   1. Поместите файл с адресами в папку 'data/'")
        print("   2. Назовите файл 'Data' (например, Data.csv или Data.xlsx)")
        print("   3. Файл должен содержать колонку 'Адрес'")
        print("\nПоддерживаемые форматы: .csv, .xlsx, .xls")
        print("\nНажмите Enter для возврата в главное меню...")
        input()
        return True

    print(f"\n✅ Найден файл: {file_path}")
    print(f"   Формат: {file_ext}")

    try:
        # Обрабатываем файл
        matcher.process_file(file_path)

        print("\n" + "=" * 60)
        print("✅ ОБОГАЩЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        print("\nРезультат сохранен в файл с суффиксом '_with_unom'")
        print("\nНажмите Enter для возврата в главное меню...")
        input()

    except Exception as e:
        print(f"\n❌ Ошибка при обработке файла: {e}")
        print("\nНажмите Enter для возврата в главное меню...")
        input()

    return True


def test_mode(matcher):
    """
    Тестовый режим - запуск диагностики
    """
    print("\n" + "=" * 50)
    print("ТЕСТОВЫЙ РЕЖИМ")
    print("=" * 50)

    # Базовые тестовые адреса
    test_queries = [
        ("Таллинская ул д 19 к 1", "Проверка Таллинской улицы"),
        ("тинао Коммунарка, поселок Завода Мосрентген, дом 33", "Проверка ТиНАО"),
        ("зеленоград корпус 1601", "Проверка ЗелАО"),
    ]

    print("\nЗапуск автоматического тестирования...")
    print("-" * 40)

    for query, description in test_queries:
        print(f"\n{description}:")
        print(f"  Запрос: {query}")
        matcher.search(query)

    print("\n" + "=" * 50)
    print("Тестирование завершено")
    print("Нажмите Enter для возврата в главное меню...")
    input()
    return True


def main():
    """Главная функция"""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Address Matcher')
    parser.add_argument('--rebuild-cache', action='store_true',
                       help='Принудительная пересборка кэша')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Папка для хранения кэша (по умолчанию: cache)')
    args = parser.parse_args()

    if args.rebuild_cache:
        print("🔧 Режим принудительной пересборки кэша")

    # Загружаем базу адресов
    print("Загрузка базы адресов...")
    try:
        df = pd.read_csv('data/addresses.csv')
        print(f"✅ Загружено {len(df)} адресов")
    except FileNotFoundError:
        print("❌ Ошибка: Файл data/addresses.csv не найден!")
        print("Убедитесь, что файл с адресами находится в папке data/")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        sys.exit(1)

    # Преобразуем УНОМ в int (если нужно)
    if 'УНОМ' in df.columns:
        df['УНОМ'] = pd.to_numeric(df['УНОМ'], errors='coerce').fillna(0).astype(int)

    # Создаем матчер
    print("\nИнициализация поискового движка...")
    matcher = AddressMatcher(df, use_index=True, cache_dir=args.cache_dir, force_rebuild=args.rebuild_cache)

    # Выводим баннер
    print_banner()

    # Основной цикл
    running = True
    while running:
        print_menu()
        choice = input("\nВаш выбор: ").strip()

        if choice == '1':
            running = interactive_mode(matcher)
        elif choice == '2':
            running = file_mode(matcher)
        elif choice == '3':
            running = test_mode(matcher)
        elif choice == '0':
            print("\nДо свидания!")
            running = False
        else:
            print("\n❌ Неверный выбор. Пожалуйста, выберите 0, 1, 2 или 3.")


if __name__ == "__main__":
    main()
