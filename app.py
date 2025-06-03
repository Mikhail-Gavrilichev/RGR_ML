import streamlit as st
import zipfile
import os
import pickle
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

@st.cache_data
def load_original_data():
    df = pd.read_csv('data_classification.csv')
    return df

original_data = load_original_data()
required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'AgeGroup']

# Сайдбар для навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти:", ["О разработчике", "Информация о наборе данных", "Визуализация", "Предсказания"])

# ... (остальные страницы без изменений)

elif page == "Предсказания":
    st.title("Предсказание сердечно-сосудистых заболеваний")
    
    # Маппинг названий моделей
    model_mapping = {
        "KNN": "knn",
        "Gradient Boosting": "gb",
        "CatBoost": "catboost",
        "Bagging": "bagging",
        "Stacking": "stacking",
        "Keras": "keras"
    }
    
    model_type = st.selectbox("Выберите модель:", list(model_mapping.keys()))
    
    # Вывод метрик качества сразу при выборе модели
    st.subheader("Метрики качества выбранной модели")
    
    model = models[model_mapping[model_type]]
    X_original = original_data[required_cols]
    y_original = original_data['num']
    y_original_binary = (y_original > 0).astype(int)
    
    # Получаем предсказания с учетом типа модели
    if model_type == "Keras":
        original_predictions = np.argmax(model.predict(X_original), axis=-1)
    else:
        original_predictions = model.predict(X_original)
    
    original_predictions_binary = (original_predictions > 0).astype(int)
    
    # Вычисляем метрики
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
    
    # Разделы для ввода данных
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
                    # Особенная обработка для Keras
                    if model_type == "Keras":
                        predictions = np.argmax(model.predict(input_df[required_cols]), axis=-1)
                        predictions_binary = (predictions > 0).astype(int)
                        
                        # Получаем вероятности для Keras
                        probas = model.predict(input_df[required_cols])
                        disease_proba = np.max(probas, axis=1)
                    else:
                        predictions = model.predict(input_df[required_cols])
                        predictions_binary = (predictions > 0).astype(int)
                        
                        if hasattr(model, "predict_proba"):
                            probas = model.predict_proba(input_df[required_cols])
                            disease_proba = probas[:, 1] if probas.shape[1] == 2 else [sum(p[1:]) for p in probas]

                    result_df = input_df.copy()
                    result_df['Prediction'] = ['Есть заболевание' if p == 1 else 'Нет заболевания' for p in predictions_binary]
                    
                    if 'disease_proba' in locals():
                        result_df['Probability'] = disease_proba

                    st.success("Предсказание завершено!")
                    st.dataframe(result_df)

                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Скачать предсказания",
                        csv,
                        "heart_disease_predictions.csv",
                        "text/csv"
                    )
            else:
                st.error(f"CSV файл должен содержать колонки: {', '.join(required_cols)}")

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
        
            # Особенная обработка для Keras
            if model_type == "Keras":
                prediction = np.argmax(model.predict(input_data), axis=-1)[0]
                proba = model.predict(input_data)
                disease_proba = np.max(proba[0])
            else:
                prediction = model.predict(input_data)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_data)
                    disease_proba = proba[0][1] if proba.shape[1] == 2 else sum(proba[0][1:])
            
            # Обработка результатов
            if prediction == 0:
                st.success("Результат: **Заболевание не обнаружено**")
            else:
                st.error(f"Результат: **Обнаружено заболевание (класс {prediction})**")
            
            if 'disease_proba' in locals():
                st.write(f"Вероятность заболевания: {disease_proba:.1%}")
                st.progress(int(disease_proba * 100))
