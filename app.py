import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Описова статистика")

dataset_name = st.selectbox("Оберіть набір даних:", ["titanic", "iris", "penguins", "Ваш файл(.CSV)"])
#dataset_name = st.selectbox("Оберіть набір даних:", sns.get_dataset_names())

if dataset_name == "titanic":
    st.info("**Titanic (Титанік):** Класичний набір даних про пасажирів сумнозвісного лайнера. Містить детальну інформацію про вік, стать, клас квитка та вартість проїзду. Найцікавіше тут — дослідити, які саме фактори найбільше вплинули на шанси людини врятуватися під час катастрофи.")
elif dataset_name == "iris":
    st.info("**Iris (Іриси):** Найвідоміший біологічний датасет у світі Data Science, створений у 1936 році! Це результати вимірювань розмірів пелюсток для трьох різних видів ірису. Ідеально показує, як статистика допомагає розрізняти види між собою.")
elif dataset_name == "penguins":
    st.info("**Penguins (Пінгвіни):** Дані, зібрані дослідниками в Антарктиці. Містять інформацію про три види пінгвінів. Ви зможете проаналізувати їхню вагу, розміри дзьобів та довжину ласт залежно від статі та острова проживання.")

if dataset_name != "Ваш файл(.CSV)":
    df = sns.load_dataset(dataset_name)
else:
    uploaded_file = st.file_uploader("Завантажте CSV файл")
    if uploaded_file is not None:
         df = pd.read_csv(uploaded_file, decimal=',')
    else:
         st.stop()

st.subheader("Перегляд даних")
#st.write(df.head())
st.dataframe(df)

st.write(f"**Кількість кортежів (рядків):** {df.shape[0]}")
st.write(f"**Кількість змінних (стовпців):** {df.shape[1]}")

st.subheader("Типи змінних:")
st.write(df.dtypes.astype(str))

st.subheader("Описова статистика (Descriptive Statistics)")
st.write(df.describe())

st.subheader("Візуалізація")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
col_to_plot = st.selectbox("Оберіть змінну для графіка:", numeric_cols)

fig, ax = plt.subplots()
sns.histplot(df[col_to_plot], kde=True, ax=ax)
st.pyplot(fig)
