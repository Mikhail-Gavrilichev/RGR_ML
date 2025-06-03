import streamlit as st
import zipfile
import os
        
st.set_page_config(layout="wide", page_title="Heart Disease Prediction")

import pickle
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from catboost import CatBoostClassifier
from tensorflow.keras.models import load_model as load_keras_model
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)

# Загрузка моделей
@st.cache_resource
def load_models():
    models = {}
    models_dir = 'models/RGR_models'
    
    with open(os.path.join(models_dir, 'RGR_model_KNN.pkl'), 'rb') as f:
        models['knn'] = pickle.load(f)
    with open(os.path.join(models_dir, 'RGR_GradientBoosting.pkl'), 'rb') as f:
        models['gb'] = pickle.load(f)
    models['catboost'] = CatBoostClassifier().load_model(os.path.join(models_dir, 'RGR_CatBoost.cbm'))
    with open(os.path.join(models_dir, 'RGR_bagging.pkl'), 'rb') as f:
        models['bagging'] = pickle.load(f)
    with open(os.path.join(models_dir, 'RGR_Stacking.pkl'), 'rb') as f:
        models['stacking'] = pickle.load(f)
    models['keras'] = load_keras_model(os.path.join(models_dir, 'RGR_Keras_Adam.h5'))
    
    return models
        
def unpack_models(archive_path='RGR_models.zip', extract_to='models'):
    try:
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_to)
        st.success("Модели успешно распакованы!")
        return True
    except Exception as e:
        st.error(f"Ошибка распаковки: {e}")
        return False

unpack_models()
models = load_models()

# Сайдбар для навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти:",
                        ["О разработчике", "Информация о наборе данных", "Визуализация", "Предсказания"])

# Страница 1: О разработчике
if page == "О разработчике":
    st.title("Информация о разработчике")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("developer_photo.jpg", width=200)
        st.write("**Имя:** Гавриличев Михаил Алексеевич")
        st.write("**Группа:** ФИТ-231")

    with col2:
        st.header("Тема РГР")
        st.write("Предсказание сердечно-сосудистых заболеваний")

# Страница 2: О наборе данных
elif page == "Информация о наборе данных":
    st.title("Информация о наборе данных по сердечным заболеваниям")

    st.header("Описание датасета")
    st.write("""
    Этот набор данных содержит медицинскую информацию о 303 пациентах и их статусе заболеваний сердца.
    Столбец 'num' - это целевая переменная (0 = нет заболевания, 1-4 = наличие заболевания).
    """)

    st.header("Описание признаков")
    features = pd.DataFrame({
        'Признаки': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'Описание': [
            'Возраст в годах',
            'Пол (1 = мужской; 0 = женский)',
            'Тип боли в груди (1-4)',
            'Артериальное давление в состоянии покоя (мм рт. ст.)',
            'Сывороточный холестерин (мг/дл)',
            'Уровень сахара в крови натощак > 120 мг/дл (1 = верно; 0 = неверно)',
            'Результаты электрокардиографии в состоянии покоя (0-2)',
            'Максимальная достигнутая частота сердечных сокращений',
            'Стенокардия, вызванная физической нагрузкой (1 = да; 0 = нет)',
            'Депрессия ST, вызванная физической нагрузкой, относительно покоя',
            'Наклон пикового сегмента ST при физической нагрузке',
            'Количество крупных сосудов (0-3), окрашенных с помощью флюороскопии',
            'Талассемия (3 = норма; 6 = фиксированный дефект; 7 = обратимый дефект)'
        ],
        'Типы': ['Numeric', 'Binary', 'Categorical', 'Numeric', 'Numeric',
                 'Binary', 'Categorical', 'Numeric', 'Binary', 'Numeric',
                 'Categorical', 'Numeric', 'Categorical']
    })
    st.table(features)

# Страница 3: Визуализации
elif page == "Визуализация":
    st.title("Анализ данных и визуализация")

    @st.cache_data
    def load_data():
        df = pd.read_csv('data_classification.csv')
        df['heart_disease'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        return df

    df = load_data()

    st.subheader("Распределение заболеваний сердца")
    fig1, ax1 = plt.subplots()
    df['heart_disease'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_xticklabels(['Нет заболевания', 'Есть заболевание'], rotation=0)
    st.pyplot(fig1)

    st.subheader("Распределение возраста по статусу заболевания")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='heart_disease', y='age', ax=ax2)
    ax2.set_xticklabels(['Нет заболевания', 'Есть заболевание'])
    st.pyplot(fig2)

# Страница 4: Предсказания
elif page == "Предсказания":
    st.title("Предсказание сердечно-сосудистых заболеваний")

    @st.cache_data
    def load_original_data():
        df = pd.read_csv('data_classification.csv')
        return df
            
    original_data = load_original_data()
    required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'AgeGroup']
        
    model_type = st.selectbox("Выберите модель:",
                        ["KNN", "GB", "CatBoost", "Bagging", "Stacking", "Keras"])
        
    input_method = st.radio("Способ ввода данных:", ["Загрузка CSV файла", "Ручной ввод"])

    if input_method == "Загрузка CSV файла":
        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            input_df['AgeGroup'] = input_df['age'].apply(
                lambda age: 1 if age <= 20 else 2 if age <= 40 else 3 if age <= 60 else 4
            )

            if all(col in input_df.columns for col in required_cols):
                st.write("Предпросмотр данных:")
                st.dataframe(input_df.head())

                if st.button("Предсказать"):
                    model = models[model_type.lower().replace(" ", "")]
                    predictions = model.predict(input_df[required_cols])
                    
                    # Преобразуем в бинарную классификацию
                    predictions_binary = (predictions > 0).astype(int)
                    
                    result_df = input_df.copy()
                    result_df['Prediction'] = ['Есть заболевание' if p == 1 else 'Нет заболевания' for p in predictions_binary]
                    
                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba(input_df[required_cols])
                        if probas.shape[1] > 2:  # Многоклассовый случай
                            result_df['Probability'] = [sum(p[1:]) for p in probas]
                        else:
                            result_df['Probability'] = probas[:, 1]

                    st.success("Предсказание завершено!")
                    st.dataframe(result_df)

                    # Оценка качества модели
                    st.subheader("Метрики качества модели")
                    
                    X_original = original_data[required_cols]
                    y_original = original_data['num']
                    y_original_binary = (y_original > 0).astype(int)
                    
                    original_predictions = model.predict(X_original)
                    original_predictions_binary = (original_predictions > 0).astype(int)
                    
                    accuracy = accuracy_score(y_original_binary, original_predictions_binary)
                    precision = precision_score(y_original_binary, original_predictions_binary)
                    recall = recall_score(y_original_binary, original_predictions_binary)
                    f1 = f1_score(y_original_binary, original_predictions_binary)
                    
                    metrics_df = pd.DataFrame({
                        'Метрика': ['Точность (Accuracy)', 'Точность (Precision)', 
                                   'Полнота (Recall)', 'F1-мера'],
                        'Значение': [f"{accuracy:.3f}", f"{precision:.3f}", 
                                    f"{recall:.3f}", f"{f1:.3f}"]
                    })
                    st.table(metrics_df)
                    
                    # Матрица ошибок
                    st.subheader("Матрица ошибок")
                    cm = confusion_matrix(y_original_binary, original_predictions_binary)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                                xticklabels=['Нет заболевания', 'Есть заболевание'],
                                yticklabels=['Нет заболевания', 'Есть заболевание'])
                    ax.set_xlabel('Предсказание')
                    ax.set_ylabel('Факт')
                    st.pyplot(fig)

    else:  # Ручной ввод
        st.subheader("Введите информацию о пациенте:")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Возраст (лет)", min_value=20, max_value=100, value=50)
            sex = st.radio("Пол", ["Мужской", "Женский"])
            cp = st.selectbox("Тип боли в груди", [
                "Типичная стенокардия",
                "Атипичная стенокардия",
                "Боль без связи со стенокардией",
                "Бессимптомная"
            ])
            trestbps = st.number_input("Давление в покое (мм рт. ст.)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Холестерин (мг/дл)", min_value=100, max_value=600, value=200)
            fbs = st.radio("Сахар натощак > 120 мг/дл", ["Нет", "Да"])

        with col2:
            restecg = st.selectbox("ЭКГ в покое", [
                "Норма",
                "ST-T аномалия",
                "Гипертрофия левого желудочка"
            ])
            thalach = st.number_input("Макс. частота сердцебиения", min_value=60, max_value=220, value=150)
            exang = st.radio("Стенокардия при нагрузке", ["Нет", "Да"])
            oldpeak = st.number_input("Депрессия ST при нагрузке", min_value=0.0, max_value=6.0, value=1.0)
            slope = st.selectbox("Наклон сегмента ST", [
                "Восходящий",
                "Плоский",
                "Нисходящий"
            ])
            ca = st.number_input("Количество сосудов", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Талассемия", [
                "Норма",
                "Фиксированный дефект",
                "Обратимый дефект"
            ])

        if st.button("Предсказать"):
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [1 if sex == "Мужской" else 0],
                'cp': [["Типичная стенокардия", "Атипичная стенокардия", 
                       "Боль без связи со стенокардией", "Бессимптомная"].index(cp) + 1],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [1 if fbs == "Да" else 0],
                'restecg': [["Норма", "ST-T аномалия", "Гипертрофия левого желудочка"].index(restecg)],
                'thalach': [thalach],
                'exang': [1 if exang == "Да" else 0],
                'oldpeak': [oldpeak],
                'slope': [["Восходящий", "Плоский", "Нисходящий"].index(slope) + 1],
                'ca': [ca],
                'thal': [["Норма", "Фиксированный дефект", "Обратимый дефект"].index(thal) + 3],
                'AgeGroup': [1 if age <= 20 else 2 if age <= 40 else 3 if age <= 60 else 4]
            })
        
            model = models[model_type.lower().replace(" ", "")]
            prediction = model.predict(input_data)[0]
            
            # Обработка результатов
            if prediction == 0:
                st.success("Результат: **Заболевание не обнаружено**")
            else:
                st.error(f"Результат: **Обнаружено заболевание (класс {prediction})**")
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)
                if proba.shape[1] > 2:  # Многоклассовый случай
                    disease_proba = sum(proba[0][1:])
                else:
                    disease_proba = proba[0][1]
                
                st.write(f"Вероятность заболевания: {disease_proba:.1%}")
                st.progress(int(disease_proba * 100))
