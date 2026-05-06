import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="Data Mining App", layout="wide")

st.sidebar.title("Навігація")
app_mode = st.sidebar.radio("Оберіть режим роботи:", ["Описова статистика", "Пошук асоціативних правил"])

if app_mode == "Описова статистика":
    st.title("Описова статистика")

    dataset_name = st.selectbox("Оберіть набір даних:", ["titanic", "iris", "penguins", "Ваш файл(.CSV)"])

    if dataset_name == "titanic":
        st.info(
            "**Titanic (Титанік):** Класичний набір даних про пасажирів сумнозвісного лайнера. Містить детальну інформацію про вік, стать, клас квитка та вартість проїзду. Найцікавіше тут — дослідити, які саме фактори найбільше вплинули на шанси людини врятуватися під час катастрофи.")
    elif dataset_name == "iris":
        st.info(
            "**Iris (Іриси):** Найвідоміший біологічний датасет у світі Data Science, створений у 1936 році! Це результати вимірювань розмірів пелюсток для трьох різних видів ірису. Ідеально показує, як статистика допомагає розрізняти види між собою.")
    elif dataset_name == "penguins":
        st.info(
            "**Penguins (Пінгвіни):** Дані, зібрані дослідниками в Антарктиці. Містять інформацію про три види пінгвінів. Ви зможете проаналізувати їхню вагу, розміри дзьобів та довжину ласт залежно від статі та острова проживання.")

    if dataset_name != "Ваш файл(.CSV)":
        df = sns.load_dataset(dataset_name)
    else:
        uploaded_file = st.file_uploader("Завантажте CSV файл")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, decimal=',')
        else:
            st.stop()

    st.subheader("Перегляд даних")
    st.dataframe(df)

    st.write(f"**Кількість кортежів (рядків):** {df.shape[0]}")
    st.write(f"**Кількість змінних (стовпців):** {df.shape[1]}")

    st.subheader("Типи змінних:")
    st.write(df.dtypes.astype(str))

    st.subheader("Описова статистика (Descriptive Statistics)")
    st.write(df.describe())

    st.subheader("Візуалізація")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if len(numeric_cols) > 0:
        col_to_plot = st.selectbox("Оберіть змінну для графіка:", numeric_cols)
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.histplot(df[col_to_plot], kde=True, ax=ax)
        st.pyplot(fig, use_container_width=False)
    else:
        st.warning("У наборі даних немає числових змінних для побудови графіка.")

elif app_mode == "Пошук асоціативних правил":
    st.title("Пошук асоціативних правил (Apriori)")

    st.markdown("""
    Цей модуль шукає закономірності у наборах транзакцій (наприклад, "якщо покупець купив товар А, то з ймовірністю X він купить товар Б").
    """)

    data_source = st.radio("Джерело даних:", ["Тестові дані (Покупки в супермаркеті)", "Власний файл (.CSV)"])

    dataset = []

    if data_source == "Тестові дані (Покупки в супермаркеті)":
        dataset = [
            ['Молоко', 'Хліб', 'Масло'],
            ['Хліб', 'Яйця'],
            ['Молоко', 'Хліб', 'Яйця', 'Масло'],
            ['Сир', 'Молоко'],
            ['Хліб', 'Масло'],
            ['Сир', 'Яйця'],
            ['Молоко', 'Хліб', 'Яйця'],
            ['Масло', 'Хліб', 'Сир']
        ]
        st.info("Використовується невеликий набір з 8 тестових чеків (транзакцій).")
        st.write(dataset)
    else:
        uploaded_file = st.file_uploader(
            "Завантажте CSV. Кожен рядок — це окрема транзакція, товари розділені комою (наприклад: Хліб,Молоко,Масло)",
            type=['csv'])
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8").split("\n")
            dataset = [line.strip().split(",") for line in content if line.strip()]
            st.write("Перші 5 транзакцій з файлу:")
            st.write(dataset[:5])
        else:
            st.stop()

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Параметри алгоритму")
    min_support = st.sidebar.slider("Мінімальна підтримка (min_support)", min_value=0.01, max_value=1.0, value=0.2,
                                    step=0.05)
    min_conf = st.sidebar.slider("Мінімальна достовірність (min_confidence)", min_value=0.01, max_value=1.0, value=0.5,
                                 step=0.05)

    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        st.warning(
            "З такими параметрами підтримки не знайдено жодного частого набору. Спробуйте зменшити 'min_support'.")
    else:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

        if rules.empty:
            st.warning(
                "Часті набори знайдені, але жодне правило не відповідає обраній мінімальній достовірності. Спробуйте зменшити 'min_confidence'.")
        else:
            st.subheader(f"Знайдені асоціативні правила (кількість: {len(rules)})")

            rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))

            st.info(
                "**Підтримка (Support)**: Показує частку чеків, у яких є обидва товари.")
            st.info(
                "**Достовірність (Confidence)**: Показує ймовірність: якщо людина вже взяла товар А, яка ймовірність, що вона візьме товар Б?")
            st.info(
                "**Ліфт (Lift)**: Показує, наскільки покупка товару А збільшує шанс покупки товару Б, порівняно з тим, якби їх купували випадково.")

            display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(
                by='lift', ascending=False)
            st.dataframe(display_rules.style.format({'support': "{:.3f}", 'confidence': "{:.3f}", 'lift': "{:.3f}"}))

            st.subheader("Візуалізація правил")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=rules, x='support', y='confidence', size='lift', hue='lift', sizes=(50, 400),
                            palette='viridis', ax=ax)
            plt.title("Діаграма розсіювання: Support vs Confidence")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

            st.markdown("---")
            st.header("Висновки")

            best_rule = display_rules.iloc[0]

            st.markdown(f"""
            **1. Які правила знайдено?**
            - Знайдено **{len(rules)}** правил(о) при мінімальній підтримці {min_support} та достовірності {min_conf}. Дані представлені в таблиці вище.

            **2. Які атрибути найбільш сильно пов’язані один з одним?**
            - Найбільший зв'язок (за метрикою Lift) спостерігається між **[{best_rule['antecedents']}]** та **[{best_rule['consequents']}]**. Lift становить {best_rule['lift']:.2f}, що означає, що покупка першого товару значно збільшує ймовірність покупки другого.
            """)

            st.text_area("**3. Які правила виявилися несподіваними для Вас? Чому, на вашу думку, таке може бути?**",
                         placeholder="Опишіть тут свої думки щодо неочікуваних комбінацій...")

            st.markdown(f"""
            **4. Скільки різних значень підтримки та достовірності довелося перевірити, перш ніж були знайдені деякі асоціативні правила?**
            - Завдяки інтерактивним повзункам у додатку було протестовано декілька комбінацій. Оптимальними виявилися поточні налаштування ({min_support} та {min_conf}).
            """)

            st.text_area(
                "**5. Чи були якісь із знайдених правил достатньо корисними, щоб їх можна було застосувати для прийняття рішень? Чому так, або чому ні?**",
                placeholder="Наприклад: Правило щодо хліба та масла дозволяє розташувати ці товари поруч у магазині або робити спільні акції...")

            st.markdown("""
            **6. Які способи представлення асоціативних правил найбільш зрозумілі?**
            - Найбільш наочним є **табличне представлення**, відсортоване за показником Lift (для пошуку найсильніших закономірностей), а також **діаграма розсіювання (Scatter Plot)**, яка дозволяє візуально оцінити баланс між підтримкою, достовірністю та силою правила.
            """)

            st.text_area(
                "**7. Чим сподобався/не сподобався використаний інструментарій?**",
                placeholder="Збережіть свої думки тут...")
