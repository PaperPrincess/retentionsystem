import sqlite3
import random
import pandas as pd
from datetime import datetime, timedelta

def init_db():
    conn = sqlite3.connect('academy_mvp.db')
    cursor = conn.cursor()
    
    # 1. Таблица студентов (заменяет PostgreSQL в MVP)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY,
            fio TEXT,
            course TEXT,
            academic_score REAL,
            risk_score REAL
        )
    ''')
    
    # 2. Таблица логов активности (заменяет ClickHouse в MVP)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lms_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            action_type TEXT,
            timestamp TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    ''')
    conn.commit()
    conn.close()

def populate_data():
    conn = sqlite3.connect('academy_mvp.db')
    cursor = conn.cursor()
    
    # Очистим таблицы перед генерацией
    cursor.execute('DELETE FROM lms_logs')
    cursor.execute('DELETE FROM students')
    
    # Списки для генерации фейковых студентов
    first_names = ["Иван", "Петр", "Сергей", "Михаил", "Алексей", "Дмитрий", "Анна", "Елена", "Мария", "Ольга"]
    last_names = ["Иванов", "Петров", "Сидоров", "Смирнов", "Кузнецов", "Попов", "Васильева", "Соколова", "Новикова"]
    courses = ["Data Science", "Web-разработка", "Тестирование ПО", "Системный анализ"]
    
    actions = ["view_lecture", "submit_assignment", "forum_post", "click_material"]
    
    print("Генерация данных...")
    for s_id in range(1001, 1051):  # Сгенерируем 50 студентов
        fio = f"{random.choice(last_names)} {random.choice(first_names)}"
        course = random.choice(courses)
        academic_score = round(random.uniform(2.5, 5.0), 2)
        
        # Экспертное правило (заглушка вместо CatBoost/LSTM): 
        # Если оценки низкие, базовый риск оттока сразу ставим выше
        risk_score = round(random.uniform(0.6, 0.95), 2) if academic_score < 3.6 else round(random.uniform(0.05, 0.5), 2)
        
        cursor.execute('INSERT INTO students VALUES (?, ?, ?, ?, ?)', 
                       (s_id, fio, course, academic_score, risk_score))
        
        # Генерируем логи активности за последние 14 дней
        num_logs = random.randint(5, 15) if academic_score < 3.6 else random.randint(30, 80) # у отличников логов больше
        base_date = datetime.now()
        
        for _ in range(num_logs):
            action = random.choice(actions)
            # Если студент в зоне риска, добавим ему случайный лог "пропуск дедлайна"
            if risk_score > 0.7 and random.random() > 0.7:
                action = "missed_deadline"
                
            days_ago = random.randint(0, 14)
            log_time = (base_date - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('INSERT INTO lms_logs (student_id, action_type, timestamp) VALUES (?, ?, ?)', 
                           (s_id, action, log_time))
            
    conn.commit()
    conn.close()
    print("База данных academy_mvp.db успешно создана и заполнена!")

if __name__ == "__main__":
    init_db()
    populate_data()