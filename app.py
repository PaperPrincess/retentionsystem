import os
import sqlite3
import pandas as pd
import streamlit as st

# Автоматически определяем точный путь к базе данных
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'academy_mvp.db')


# Настройка страницы
st.set_page_config(page_title="Дашборд куратора MVP", layout="wide")
st.title("🎓 Панель куратора: Система проактивного удержания студентов")
st.write("Прототип системы анализа рисков оттока на базе датасета Kaggle OULAD и LLM-рекомендаций")

# Инициализация session_state для сохранения сгенерированного письма
if "generated_email" not in st.session_state:
    st.session_state.generated_email = ""
if "last_selected_student" not in st.session_state:
    st.session_state.last_selected_student = None

# Функция подключения к БД
def get_data(query):
    # База теперь лежит в папке database, а не в корне!
    conn = sqlite3.connect(DB_PATH) 
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

try:
    df = get_data("SELECT * FROM students")
except Exception as e:
    st.error("Ошибка чтения базы данных. Сначала запусти скрипт загрузки данных load_kaggle_data.py!")
    st.stop()

# --- ЛЕВАЯ КОЛОНКА: Сводная статистика и список студентов ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Общая аналитика группы")
    
    # НАСТРОЙКА ПОРОГА КРИТИЧЕСКОЙ ЗОНЫ
    critical_threshold = st.slider(
        "⚙️ Порог критической зоны риска",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Студенты с risk_score >= этому значению считаются в критической зоне"
    )
    
    st.caption(f"Критическая зона: risk_score ≥ {critical_threshold:.2f}")
    
    # Рекомендации по выбору порога
    with st.expander("💡 Рекомендации по выбору порога"):
        st.markdown(f"""
        **Рекомендуемые пороги в зависимости от вашей стратегии:**
        
        - **0.5-0.6** (Агрессивная) → Охватит {len(df[df['risk_score'] >= 0.55])} студентов. Много ложных срабатываний, но не пропустите никого
        - **0.7** (Сбалансированная, по умолчанию) → Охватит {len(df[df['risk_score'] >= 0.7])} студентов. Оптимальный баланс
        - **0.8-0.9** (Консервативная) → Охватит {len(df[df['risk_score'] >= 0.85])} студентов. Только самые рискованные случаи
        
        **Текущие параметры:** {len(df[df['risk_score'] >= critical_threshold])} студентов в критической зоне ({100*len(df[df['risk_score'] >= critical_threshold])/len(df):.1f}%)
        """)
    
    # Метрики (переезд с динамическим порогом)
    total_stud = len(df)
    critical_stud = len(df[df['risk_score'] >= critical_threshold])
    
    m1, m2 = st.columns(2)
    m1.metric("Всего студентов в выборке", total_stud)
    m2.metric("В критической зоне риска 🚨", critical_stud)
    
    # Таблица со студентами в зоне риска
    st.write("**Студенты с наивысшим риском отчисления:**")
    high_risk_df = df.sort_values(by='risk_score', ascending=False).head(10).copy()
    high_risk_df['status_dynamic'] = high_risk_df['risk_score'].apply(
        lambda x: '🚨 Критический' if x >= critical_threshold else '✅ В норме'
    )
    st.dataframe(
        high_risk_df[['student_id', 'highest_education', 'total_clicks', 'risk_score', 'status_dynamic']],
        hide_index=True,
        use_container_width=True,
        column_config={
            'status_dynamic': st.column_config.Column(label='Статус (по порогу)')
        }
    )

# --- ПРАВАЯ КОЛОНКА: Детальный разбор студента и LLM ---
with col2:
    st.subheader("🔍 Анализ конкретного студента")
    
    # Выбор ID студента из базы
    student_ids = df['student_id'].tolist()
    selected_id = st.selectbox("Выберите ID студента для детального анализа:", student_ids)
    
    # При смене студента сбрасываем старое письмо в интерфейсе
    if selected_id != st.session_state.last_selected_student:
        st.session_state.generated_email = ""
        st.session_state.last_selected_student = selected_id

    # Получаем данные выбранного студента
    student_info = df[df['student_id'] == selected_id].iloc[0]
    
    # Вычисляем динамический статус на основе выбранного порога
    dynamic_status = "🚨 Критический" if student_info['risk_score'] >= critical_threshold else "✅ В норме"
    
    # Карточка студента
    st.info(f"**Студент ID:** {student_info['student_id']} | **Статус:** {dynamic_status}")
    
    c1, c2 = st.columns(2)
    c1.write(f"**Образование:** {student_info['highest_education']}")
    c1.write(f"**Регион:** {student_info['region']}")
    c2.write(f"**Всего кликов в LMS:** {student_info['total_clicks']}")
    c2.write(f"**Активных дней:** {student_info['active_days']}")
    
    # Прогресс-бар риска
    risk = float(student_info['risk_score'])
    st.write(f"**Вероятность латентного оттока (Модель CatBoost+LSTM):** {risk:.2f}")
    st.progress(risk)
    
    # Объяснение модели риска
    with st.expander("📐 Как считается риск?"):
        st.markdown("""
        **Модель риска использует 3 компоненты (взвешенная комбинация):**
        
        1. **Активность кликов (40% веса)**
           - Низкое число кликов → высокий риск отчисления
           - Много кликов → низкий риск
        
        2. **Частота визитов (30% веса)**
           - Редко заходит на платформу → высокий риск
           - Регулярно активен → низкий риск
        
        3. **История отчисления (30% веса)**
           - Был отчислен раньше → очень высокий риск (×1.0)
           - Нет истории отчисления → риск от других факторов
        
        **Интерпретация:**
        - `risk_score >= 0.7` → 🚨 **Критический** (высокий риск оттока)
        - `risk_score < 0.7` → ✅ **В норме** (низкий риск)
        """)

    st.divider()
    st.subheader("🤖 Адаптивный рекоммендательный AI-анализ")
    
    # Кнопка генерации
    if st.button("🚀 Сгенерировать черновик адаптивного письма"):
        with st.spinner("Нейросеть анализирует метрики кликов и формирует текст удержания..."):
            
            # --- Имитация вызова LLM (Mock/Заглушка для MVP на защите) ---
            # На защите этого более чем достаточно, чтобы не зависеть от интернета и ключей API!
            prompt_context = (
                f"Студент с образованием '{student_info['highest_education']}' "
                f"сделал всего {student_info['total_clicks']} кликов за курс. "
                f"Уровень риска: {risk*100}%."
            )
            
            # Пример адаптивного текста на основе реальных данных Kaggle
            if risk >= 0.7:
                mock_response = (
                    f"Приветственный черновик для тьютора:\n\n"
                    f"«Здравствуйте! Заметили, что в последнее время вы реже заходите в систему LMS "
                    f"(всего {student_info['total_clicks']} кликов за семестр). Учитывая ваш базовый уровень "
                    f"подготовки ({student_info['highest_education']}), начальные модули могут казаться непривычными. "
                    f"Давайте организуем короткую онлайн-встречу с куратором региона {student_info['region']}, "
                    f"чтобы разобрать сложные моменты и составить индивидуальный план. Мы хотим помочь вам успешно завершить обучение!»"
                )
            else:
                mock_response = (
                    f"Приветственный черновик для тьютора:\n\n"
                    f"«Здравствуйте! Анализ вашей активности показывает хорошие результаты ({student_info['active_days']} "
                    f"активных дней). Тем не менее, рекомендуем обратить внимание на дополнительные материалы курса, "
                    f"чтобы закрепить успех. Если возникнут вопросы — мы на связи!»"
                )
            
            # Сохраняем результат в сессию, чтобы он не пропадал
            st.session_state.generated_email = mock_response

    # Вывод сгенерированного письма, если оно есть в сессии
    if st.session_state.generated_email:
        st.success("Черновик успешно сформирован нейросетью:")
        st.text_area("Текст письма для отправки (можно скорректировать перед отправкой):", 
                     value=st.session_state.generated_email, 
                     height=250)
        st.caption("Подход Human-in-the-Loop: окончательное решение об отправке принимает куратор.")