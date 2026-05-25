"""
Модуль для пакетной обработки файлов с адресами
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.matcher import AddressMatcher


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


def run_file_mode(matcher: AddressMatcher) -> bool:
    """
    Запуск файлового режима обработки адресов.
    Args:
        matcher: Экземпляр AddressMatcher
    Returns:
        bool: True для возврата в главное меню
    """
    print("\n" + "=" * 60)
    print("ФАЙЛОВЫЙ РЕЖИМ ОБРАБОТКИ АДРЕСОВ")
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
        print("✅ ОБРАБОТКА ЗАВЕРШЕНА")
        print("=" * 60)
        print("\nРезультат сохранен в файл с суффиксом '_with_unom'")
        print("\nНажмите Enter для возврата в главное меню...")
        input()

    except Exception as e:
        print(f"\n❌ Ошибка при обработке файла: {e}")
        print("\nНажмите Enter для возврата в главное меню...")
        input()

    return True
