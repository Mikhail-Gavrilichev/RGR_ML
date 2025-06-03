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

# Загрузка моделей
@st.cache_resource
def load_models():
    models = {}
    models_dir = 'models/RGR_models'  # обновлено
    
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
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:",
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
    Столбец «num» — это целевая переменная (0 = нет заболевания, 1–4 = наличие заболевания).
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

    st.header("Обработка данных")
    st.write("""
    - Изменение типа данных: Тип данных столбцов 'ca' и 'thal' изменен на int.
    - Редактирование данных: Добавлены возрастные группы
    - Обработка выбросов: Выбросы обнаруженые в параметрах chol, oldpeak удалены.
    """)

# Страница 3: Визуализации
elif page == "Визуализация":
    st.title("Анализ данных и визуализация")


    @st.cache_data
    def load_data():
        df = pd.read_csv('data_classification.csv')
        # Create binary target for visualization
        df['heart_disease'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        return df


    df = load_data()

    # 1. Target distribution
    st.subheader("Heart Disease Distribution")
    fig1, ax1 = plt.subplots()
    df['heart_disease'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_xticklabels(['No Disease', 'Disease'], rotation=0)
    st.pyplot(fig1)

    # 2. Age distribution by disease status
    st.subheader("Age Distribution by Disease Status")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='heart_disease', y='age', ax=ax2)
    ax2.set_xticklabels(['No Disease', 'Disease'])
    st.pyplot(fig2)

    # 3. Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    sns.heatmap(df[numeric_cols + ['heart_disease']].corr(), annot=True, ax=ax3)
    st.pyplot(fig3)

    # 4. Cholesterol vs Age with disease status
    st.subheader("Cholesterol vs Age by Disease Status")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x='age', y='chol', hue='heart_disease', ax=ax4)
    ax4.legend(title='Heart Disease', labels=['No', 'Yes'])
    st.pyplot(fig4)

# Страница 4: Предсказания
elif page == "Предсказания":
    st.title("Предсказания сердечно-сосудистых заболеваний")

    # Загрузка оригинальных данных для оценки качества
    @st.cache_data
    def load_original_data():
        df = pd.read_csv('data_classification.csv')
        return df

    original_data = load_original_data()
    required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'AgeGroup']

    # Выбор модели
    model_type = st.selectbox("Select Model:",
                            ["KNN", "gb", "CatBoost", "Bagging", "Stacking", "Keras"])

    # Способ ввода данных
    input_method = st.radio("Input Method:", ["Upload CSV File", "Manual Input"])

    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            input_df['AgeGroup'] = input_df['age'].apply(
                    lambda age: 1 if age <= 20 else 2 if age <= 40 else 3 if age <= 60 else 4
                )

            if all(col in input_df.columns for col in required_cols):
                st.write("Uploaded Data Preview:")
                st.dataframe(input_df.head())

                if st.button("Predict"):
                    model = models[model_type.lower().replace(" ", "")]
                    
                    # Получаем предсказания
                    predictions = model.predict(input_df[required_cols])
                    
                    # Получаем вероятности, если модель поддерживает
                    proba = model.predict_proba(input_df[required_cols])[:, 1] if hasattr(model,
                                                                                        "predict_proba") else None

                    result_df = input_df.copy()
                    result_df['Prediction'] = ['Disease' if p == 1 else 'No Disease' for p in predictions]
                    if proba is not None:
                        result_df['Probability'] = proba

                    st.success("Predictions completed!")
                    st.dataframe(result_df)

                    # Вычисляем метрики качества на оригинальных данных
                    st.subheader("Model Quality Metrics (на оригинальных данных)")
                    
                    # Подготовка данных
                    X_original = original_data[required_cols]
                    y_original = original_data['num']
                    
                    # Получаем предсказания для оригинальных данных
                    original_predictions = model.predict(X_original)
                    
                    # Вычисляем метрики
                    from sklearn.metrics import (accuracy_score, precision_score, 
                                              recall_score, f1_score, roc_auc_score,
                                              confusion_matrix, classification_report)
                    
                    accuracy = accuracy_score(y_original, original_predictions)
                    precision = precision_score(y_original, original_predictions)
                    recall = recall_score(y_original, original_predictions)
                    f1 = f1_score(y_original, original_predictions)
                    
                    # ROC-AUC (только если модель возвращает вероятности)
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_original)[:, 1]
                        roc_auc = roc_auc_score(y_original, y_proba)
                    else:
                        roc_auc = "N/A"
                    
                    # Отображаем метрики
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        'Value': [f"{accuracy:.3f}", f"{precision:.3f}", 
                                 f"{recall:.3f}", f"{f1:.3f}", 
                                 f"{roc_auc:.3f}" if isinstance(roc_auc, float) else roc_auc]
                    })
                    
                    st.table(metrics_df)
                    
                    # Матрица ошибок
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_original, original_predictions)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax, 
                                xticklabels=['No Disease', 'Disease'], 
                                yticklabels=['No Disease', 'Disease'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    
                    # Отчет классификации
                    st.subheader("Classification Report")
                    report = classification_report(y_original, original_predictions, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.table(report_df)

                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "heart_disease_predictions.csv",
                        "text/csv"
                    )
            else:
                st.error(f"CSV file must contain these columns: {', '.join(required_cols)}")

    else:  # Ручной ввод
        st.subheader("Введите информацию о пациенте:")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
            sex = st.radio("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", [
                "Typical angina",
                "Atypical angina",
                "Non-anginal pain",
                "Asymptomatic"
            ])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

        with col2:
            restecg = st.selectbox("Resting ECG", [
                "Normal",
                "ST-T wave abnormality",
                "Left ventricular hypertrophy"
            ])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                "Upsloping",
                "Flat",
                "Downsloping"
            ])
            ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thalassemia", [
                "Normal",
                "Fixed Defect",
                "Reversible Defect"
            ])

        if st.button("Predict"):
            # Преобразование введенных данных
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [1 if sex == "Male" else 0],
                'cp': [["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp) + 1],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [1 if fbs == "Yes" else 0],
                'restecg': [["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)],
                'thalach': [thalach],
                'exang': [1 if exang == "Yes" else 0],
                'oldpeak': [oldpeak],
                'slope': [["Upsloping", "Flat", "Downsloping"].index(slope) + 1],
                'ca': [ca],
                'thal': [["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 3],
                'AgeGroup': [1 if age <= 20 else 2 if age <= 40 else 3 if age <= 60 else 4]
            })
        
            model = models[model_type.lower().replace(" ", "")]
            predictions = model.predict(input_data)
            
            # Обработка предсказаний в зависимости от типа модели
            if model_type.lower() == "keras":
                prediction = np.argmax(predictions, axis=-1)[0]
                disease_proba = predictions[0][1:].sum()  # Сумма вероятностей классов 1-4
            else:
                prediction = predictions[0] if isinstance(predictions, (np.ndarray, list)) else predictions
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(input_data)
                    disease_proba = sum(probas[0][1:]) if len(probas[0]) > 2 else probas[0][1]
                else:
                    disease_proba = None
        
            # Вывод результатов
            if prediction == 0:
                st.success("Результат: **Болезнь не обнаружена** (класс 0)")
            else:
                st.error(f"Результат: **Обнаружена болезнь** (класс {prediction})")
        
            if disease_proba is not None:
                st.write(f"Вероятность наличия болезни (классы 1-4): {disease_proba:.1%}")
                st.progress(int(disease_proba * 100))
                
            # Вывод информации о качестве модели на оригинальных данных
            st.subheader("Model Quality Metrics (на оригинальных данных)")
            
            # Подготовка данных
            X_original = original_data[required_cols]
            y_original = original_data['num']
            
            # Получаем предсказания для оригинальных данных
            original_predictions = model.predict(X_original)
            
            # Вычисляем метрики
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_original, original_predictions)
            precision = precision_score(y_original, original_predictions)
            recall = recall_score(y_original, original_predictions)
            f1 = f1_score(y_original, original_predictions)
            
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [f"{accuracy:.3f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]
            })
            
            st.table(metrics_df)
            
            # Краткая интерпретация метрик
            st.subheader("Интерпретация метрик")
            st.write("""
            - **Accuracy (Точность)**: Доля правильных предсказаний среди всех сделанных.
            - **Precision (Точность)**: Доля правильно предсказанных больных среди всех предсказанных как больные.
            - **Recall (Полнота)**: Доль правильно предсказанных больных среди всех действительно больных.
            - **F1-Score**: Гармоническое среднее точности и полноты.
            """)
