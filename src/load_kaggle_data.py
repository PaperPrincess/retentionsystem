import os
import sqlite3
import pandas as pd

# 1. Автоматическое и динамическое определение путей относительно структуры проекта
# os.path.abspath(__file__) находит путь до самого этого скрипта (в папке src)
# Нам нужно подняться на один уровень выше, чтобы попасть в корень проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Собираем правильные пути к файлам в папке data
student_info_path = os.path.join(BASE_DIR, 'data', 'studentInfo.csv')
student_vle_path = os.path.join(BASE_DIR, 'data', 'studentVle.csv')

# Собираем путь к базе данных в папке database
db_path = os.path.join(BASE_DIR, 'database', 'academy_mvp.db')

print("=== Проверка путей ===")
print(f"Корень проекта: {BASE_DIR}")
print(f"Ищем studentInfo по пути: {student_info_path}")
print(f"Ищем studentVle по пути: {student_vle_path}")
print(f"База данных будет создана по пути: {db_path}\n")

# Проверяем физическое наличие CSV файлов перед стартом, чтобы не падать с невнятной ошибкой
if not os.path.exists(student_info_path):
    raise FileNotFoundError(f"Критическая ошибка: Файл studentInfo.csv не найден! Проверь, лежит ли он в {os.path.join(BASE_DIR, 'data')}")
if not os.path.exists(student_vle_path):
    raise FileNotFoundError(f"Критическая ошибка: Файл studentVle.csv не найден! Проверь, лежит ли он в {os.path.join(BASE_DIR, 'data')}")

# 2. Загрузка данных из CSV-файлов Kaggle
print("Загрузка данных из CSV-файлов Kaggle...")
print("  > Загружаю studentInfo.csv...")
df_info = pd.read_csv(student_info_path)
print(f"  OK. Загружено {len(df_info)} студентов")

print("  > Загружаю studentVle.csv (это займет 30-60 сек)...")
# Читаем только нужные столбцы для оптимизации памяти
df_vle = pd.read_csv(student_vle_path, usecols=['id_student', 'date', 'sum_click'])
print(f"  OK. Загружено {len(df_vle)} записей активности")

# Переводим целевой таргет в бинарный вид: 1 — если отчислен (Withdrawn), 0 — иначе
df_info['is_dropout'] = df_info['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)

# 3. Агрегируем логи активности из studentVle
print("Агрегация логов активности студентов...")
df_activity = df_vle.groupby('id_student').agg(
    total_clicks=('sum_click', 'sum'),
    active_days=('date', 'nunique')
).reset_index()
print(f"  OK. Агрегировано {len(df_activity)} студентов")

# 4. Объединяем профили студентов и их агрегированную активность
print("Объединение таблиц...")
merged_data = pd.merge(df_info, df_activity, on='id_student', how='left')
# Если у студента вообще нет кликов, заменим NaN на 0
merged_data['total_clicks'] = merged_data['total_clicks'].fillna(0)
merged_data['active_days'] = merged_data['active_days'].fillna(0)

# Дедублицируем по id_student (берём первый записанный случай)
merged_data = merged_data.drop_duplicates(subset=['id_student'], keep='first')
print(f"  OK. После дедупликации: {len(merged_data)} уникальных студентов")

# 5. Инициализация SQLite базы данных
print("Подключение к SQLite и создание структуры таблиц...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Очищаем старые данные: сначала удаляем таблицу логов (она ссылается на students), потом students
cursor.execute('DROP TABLE IF EXISTS lms_logs')
cursor.execute('DROP TABLE IF EXISTS students')
conn.commit()

# Создаем таблицу студентов в соответствии с полями из Kaggle OULAD
cursor.execute('''
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    gender TEXT,
    region TEXT,
    highest_education TEXT,
    total_clicks INTEGER,
    active_days INTEGER,
    risk_score REAL,
    status TEXT
)
''')

# Создаем таблицу логов для детального просмотра в дашборде
cursor.execute('''
CREATE TABLE lms_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    action_type TEXT,
    timestamp TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
)
''')
conn.commit()

print("Запись подготовленных данных в базу данных...")
print("  Модель риска: weighted combination of (clicks, active_days, dropout_history)\n")

# 6. Расчет рисков и заполнение базы данных
max_clicks = merged_data['total_clicks'].max() if merged_data['total_clicks'].max() > 0 else 1
max_active_days = merged_data['active_days'].max() if merged_data['active_days'].max() > 0 else 1

# Ограничимся первыми 1000 студентов, чтобы MVP работал быстро
total_students = min(1000, len(merged_data))
for idx, (_, row) in enumerate(merged_data.head(1000).iterrows()):
    if (idx + 1) % 100 == 0:
        print(f"  Записано {idx + 1}/{total_students} студентов...")
    
    # === УЛУЧШЕННАЯ МОДЕЛЬ РИСКА (3 компоненты) ===
    
    # 1. Риск на основе КЛИКОВ (40% веса)
    clicks_ratio = row['total_clicks'] / max_clicks
    clicks_risk = 1.0 - clicks_ratio  # Мало кликов = высокий риск
    
    # 2. Риск на основе АКТИВНЫХ ДНЕЙ (30% веса)
    active_days_ratio = row['active_days'] / max_active_days
    activity_risk = 1.0 - active_days_ratio  # Редко заходит = высокий риск
    
    # 3. Риск на основе ИСТОРИИ ОТЧИСЛЕНИЯ (30% веса)
    dropout_risk = 1.0 if row['is_dropout'] == 1 else 0.0  # Был отчислен = очень высокий риск
    
    # Взвешенная комбинация
    weights = {'clicks': 0.4, 'activity': 0.3, 'dropout': 0.3}
    calculated_score = round(
        weights['clicks'] * clicks_risk + 
        weights['activity'] * activity_risk + 
        weights['dropout'] * dropout_risk,
        2
    )
    
    # Ограничиваем диапазон [0, 1]
    calculated_score = max(0.0, min(1.0, calculated_score))

    status = "Критический" if calculated_score >= 0.7 else "В норме"

    # Записываем студента
    cursor.execute('''
    INSERT INTO students (student_id, gender, region, highest_education, total_clicks, active_days, risk_score, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        int(row['id_student']),
        str(row['gender']),
        str(row['region']),
        str(row['highest_education']),
        int(row['total_clicks']),
        int(row['active_days']),
        float(calculated_score),
        str(status)
    ))
    
    # Генерируем несколько логов активности для демонстрации графиков в Streamlit
    # Для студентов из зоны риска добавим "missed_deadline"
    s_id = int(row['id_student'])
    if calculated_score >= 0.7:
        cursor.execute("INSERT INTO lms_logs (student_id, action_type, timestamp) VALUES (?, ?, ?)", (s_id, 'missed_deadline', '2026-07-09 12:00:00'))
        cursor.execute("INSERT INTO lms_logs (student_id, action_type, timestamp) VALUES (?, ?, ?)", (s_id, 'click_material', '2026-07-01 10:15:21'))
    else:
        cursor.execute("INSERT INTO lms_logs (student_id, action_type, timestamp) VALUES (?, ?, ?)", (s_id, 'view_lecture', '2026-07-11 15:30:00'))
        cursor.execute("INSERT INTO lms_logs (student_id, action_type, timestamp) VALUES (?, ?, ?)", (s_id, 'submit_assignment', '2026-07-10 18:45:12'))
        cursor.execute("INSERT INTO lms_logs (student_id, action_type, timestamp) VALUES (?, ?, ?)", (s_id, 'forum_post', '2026-07-07 09:11:00'))

    if (idx + 1) % 100 == 0:
        conn.commit()

conn.commit()
conn.close()

print("")
print("BASE DATA LOAD COMPLETED SUCCESSFULLY!")
print(f"Database saved to: {db_path}")