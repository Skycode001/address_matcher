#!/usr/bin/env python3
"""
Address Matcher - система поиска адресов
"""

import os
import sys

import pandas as pd

# Добавляем путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.matcher import AddressMatcher


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
    print("   2. Файловый режим (обработка файла с адресами) [в разработке]")
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


def file_mode(matcher):
    """
    Файловый режим - обработка файла с адресами (задел на будущее)
    """
    print("\n" + "=" * 50)
    print("ФАЙЛОВЫЙ РЕЖИМ (в разработке)")
    print("=" * 50)
    print("Функционал будет добавлен в следующих версиях.")
    print("Поддерживаемые форматы: CSV, Excel, TXT")
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

    # Создаем матчер
    print("\nИнициализация поискового движка...")
    matcher = AddressMatcher(df, use_index=True)

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
