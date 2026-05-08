import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

st.set_page_config(page_title="Data Mining App", layout="wide")

st.sidebar.title("Навігація")
app_mode = st.sidebar.radio("Оберіть режим роботи:", [
    "Описова статистика",
    "Пошук асоціативних правил",
    "Комплексний кластерний аналіз"
])

if app_mode == "Описова статистика":
    st.title("Описова статистика")

    dataset_name = st.selectbox("**Оберіть набір даних:**", ["titanic", "iris", "penguins", "Ваш файл(.CSV)"])

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
        col_to_plot = st.selectbox("**Змінну для графіка:**", numeric_cols)
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.histplot(df[col_to_plot], kde=True, ax=ax)
        st.pyplot(fig, use_container_width=False)
    else:
        st.warning("У наборі даних немає числових змінних для побудови графіка.")

elif app_mode == "Пошук асоціативних правил":
    st.title("Пошук асоціативних правил (Apriori)")

    st.markdown("""
    Цей модуль шукає закономірності у наборах транзакцій.
    Наприклад, "якщо покупець купив товар А, то з ймовірністю X він купить товар Б".
    """)

    data_source = st.radio("**Джерело даних:**", ["Тестові дані (Покупки в супермаркеті)", "Власний файл (.CSV)"])

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
            st.subheader("Висновки")

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

elif app_mode == "Комплексний кластерний аналіз":
    st.title("Комплексний кластерний аналіз")

    st.markdown("""
    Цей модуль об'єднує найбільш схожі об'єкти у групи (кластери) крок за кроком, будуючи деревоподібну структуру — **дендрограму**.
    """)

    data_source = st.radio("**Джерело даних:**", ["Відкриті дані Світового банку (World Bank Open Data)", "Власний файл (.CSV)"])
    if data_source == "Відкриті дані Світового банку (World Bank Open Data)":
        st.subheader("Опис набору даних")
        st.info("""
                **Джерело даних:** Синтетична вибірка на основі відкритих даних Світового банку (World Bank Open Data).
                **Кількість об'єктів:** 20 (країни Європи).
                **Кількість змінних:** 5 числових показників (ВВП на душу, Тривалість життя, Рівень безробіття, Індекс освіти, Витрати на охорону здоров'я).
                """)

        data = {
            'Країна': ['Норвегія', 'Швейцарія', 'Німеччина', 'Франція', 'Італія', 'Іспанія', 'Греція', 'Польща',
                       'Україна', 'Румунія',
                       'Швеція', 'Фінляндія', 'Данія', 'Нідерланди', 'Бельгія', 'Австрія', 'Чехія', 'Угорщина',
                       'Португалія', 'Словаччина'],
            'ВВП на душу ($)': [89000, 92000, 51000, 43000, 35000, 30000, 20000, 17000, 4000, 15000, 54000, 53000,
                                61000, 53000, 51000, 53000, 25000, 18000, 24000, 21000],
            'Тривалість життя': [82.4, 83.8, 81.3, 82.7, 83.5, 83.6, 82.2, 78.7, 72.1, 76.0, 82.8, 81.9, 81.6, 82.2,
                                 81.6, 81.5, 79.3, 76.9, 82.0, 77.5],
            'Рівень безробіття (%)': [3.8, 4.2, 3.1, 7.3, 9.5, 13.3, 14.8, 3.2, 9.8, 5.0, 8.3, 6.7, 5.1, 3.8, 5.9, 5.1,
                                      2.8, 4.1, 6.6, 6.5],
            'Індекс освіти': [0.92, 0.90, 0.94, 0.89, 0.88, 0.87, 0.87, 0.88, 0.80, 0.81, 0.93, 0.93, 0.95, 0.91, 0.89,
                              0.88, 0.89, 0.85, 0.84, 0.86],
            'Витрати на здоров\'я (%)': [10.5, 11.3, 11.7, 11.1, 8.7, 9.0, 7.8, 6.5, 7.1, 5.7, 10.9, 9.2, 10.1, 10.0,
                                         10.3, 10.4, 7.8, 6.4, 9.5, 7.0]
        }
        df_cluster = pd.DataFrame(data)
    else:
        uploaded_file = st.file_uploader(
            "Завантажте CSV. Перший стовпець має містити назви об'єктів (наприклад, країн чи клієнтів), а всі наступні — числові показники.",
            type=['csv'])
        if uploaded_file is not None:
            df_cluster = pd.read_csv(uploaded_file)
        else:
            st.stop()

        st.subheader("Опис набору даних")
        st.text_area("**Опишіть набір даних: джерело даних (посилання), кількість об'єктів у вибірці, кількість змінних, типи змінних тощо.**",
                     placeholder="...")

    st.markdown("---")
    st.header("1. Ієрархічний кластерний аналіз")

    st.dataframe(df_cluster)
    st.markdown(
        "Згідно з правилами кластеризації, усі дані перед аналізом стандартизовані (Z-перетворення), щоб виключити вплив різних масштабів змінних.")
    scaler = StandardScaler()

    numeric_features = df_cluster.iloc[:, 1:].select_dtypes(include=['float64', 'int64'])
    scaled_features = scaler.fit_transform(numeric_features)

    labels_for_dendrogram = df_cluster.iloc[:, 0].astype(str).values

    st.markdown("---")
    st.subheader("Налаштування параметрів та Дендрограми")
    st.write("**Оберіть два різні методи для порівняння.**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Метод 1")
        metric_1 = st.selectbox("Міра відстані 1:", ["euclidean (Евклідова)", "cityblock (Манхеттенська)"], key="m1")
        linkage_1 = st.selectbox("Спосіб об'єднання 1:", ["ward (Метод Варда)", "complete (Найвіддаленішого сусіда)",
                                                          "single (Найближчого сусіда)",
                                                          "average (Середнього зв'язку)"], key="l1")

        m1_val = metric_1.split(" ")[0]
        l1_val = linkage_1.split(" ")[0]

        if l1_val == 'ward' and m1_val != 'euclidean':
            st.error("Метод Варда вимагає Евклідової відстані. Автоматично застосовано 'euclidean'.")
            m1_val = 'euclidean'

        Z1 = linkage(scaled_features, method=l1_val, metric=m1_val)

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        dendrogram(Z1, labels=labels_for_dendrogram, leaf_rotation=90, ax=ax1)
        ax1.set_title(f"Дендрограма 1 ({l1_val}, {m1_val})")
        ax1.set_ylabel("Відстань (Distance)")
        st.pyplot(fig1)

    with col2:
        st.subheader("Метод 2")
        metric_2 = st.selectbox("Міра відстані 2:", ["cityblock (Манхеттенська)", "euclidean (Евклідова)"], key="m2")
        linkage_2 = st.selectbox("Спосіб об'єднання 2:", ["complete (Найвіддаленішого сусіда)", "ward (Метод Варда)",
                                                          "single (Найближчого сусіда)",
                                                          "average (Середнього зв'язку)"], key="l2")

        m2_val = metric_2.split(" ")[0]
        l2_val = linkage_2.split(" ")[0]

        if l2_val == 'ward' and m2_val != 'euclidean':
            st.error("Метод Варда вимагає Евклідової відстані. Автоматично застосовано 'euclidean'.")
            m2_val = 'euclidean'

        Z2 = linkage(scaled_features, method=l2_val, metric=m2_val)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        dendrogram(Z2, labels=labels_for_dendrogram, leaf_rotation=90, ax=ax2)
        ax2.set_title(f"Дендрограма 2 ({l2_val}, {m2_val})")
        ax2.set_ylabel("Відстань (Distance)")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Аналіз та Висновки")

    st.markdown("""
    **1. Як визначити оптимальну кількість кластерів по дендрограмі?**
    - Знайдіть найдовші вертикальні лінії, які не перетинаються жодними горизонтальними лініями. Якщо провести уявну горизонтальну лінію через ці "довгі" вертикалі, кількість перетинів покаже оптимальну кількість кластерів.
    """)

    st.text_area("**2. Яку оптимальну кількість кластерів Ви визначили за допомогою першої та другої дендрограми?**",
                 placeholder="Наприклад: За методом Варда чітко виділяються 3 кластери (країни з високим, середнім та низьким рівнем розвитку). За методом найближчого сусіда кластери менш виражені...")

    st.text_area("**3. Які об'єкти (країни) потрапили до кожного кластеру?**",
                 placeholder="Наприклад: До першого кластеру стабільно потрапляють Норвегія та Швейцарія (дуже високий ВВП). До іншого — Україна та Румунія (нижчі економічні показники)...")

    st.text_area("**4. Наскільки стійкими виявилися кластери при застосуванні різних методів?**",
                 placeholder="Наприклад: Ядро кластерів залишається незмінним при обох методах (наприклад, скандинавські країни завжди разом). Проте метод найближчого сусіда схильний приєднувати об'єкти по одному, утворюючи 'ланцюжки', тоді як метод Варда формує більш компактні та логічні групи.")


    st.markdown("---")
    st.header("2. Ітеративний кластерний аналіз k-середніх(k-means)")

    st.markdown("""
        На відміну від ієрархічного методу, алгоритм k-середніх намагається розбити дані на заздалегідь визначену кількість кластерів так, щоб об'єкти всередині групи були максимально схожі між собою[cite: 86, 88].
        """)

    st.subheader("Визначення оптимального k (Метод ліктя)")
    wcss = []
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans_temp.fit(scaled_features)
        wcss.append(kmeans_temp.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
    ax_elbow.plot(range(1, 11), wcss, marker='o', linestyle='--')
    ax_elbow.set_title('Метод ліктя (Elbow Method)')
    ax_elbow.set_xlabel('Кількість кластерів (k)')
    ax_elbow.set_ylabel('WCSS (Внутрішньокластерна сума квадратів)')
    st.pyplot(fig_elbow)
    st.info("Порада: Оптимальне k знаходиться в точці, де графік 'згинається' (нагадує лікоть).")

    k_choice = st.slider("**Оберіть кількість кластерів (k):**", min_value=2, max_value=6, value=3)
    kmeans = KMeans(n_clusters=k_choice, init='k-means++', random_state=42, n_init=10)
    clusters_km = kmeans.fit_predict(scaled_features)

    df_cluster['Кластер (K-Means)'] = clusters_km
    st.subheader(f"Результати розподілу (k={k_choice})")
    st.dataframe(df_cluster)

    st.subheader("Профілі кластерів (Характеристики)")
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    feature_names = df_cluster.iloc[:, 1:].select_dtypes(include=['number']).columns[:centers.shape[1]]
    df_centers = pd.DataFrame(centers, columns=feature_names)
    df_centers['Кластер'] = [f'Кластер {i}' for i in range(k_choice)]

    fig_profile, ax_profile = plt.subplots(figsize=(8, 4))
    for i in range(k_choice):
        ax_profile.plot(feature_names, centers[i], marker='s', label=f'Кластер {i}')

    ax_profile.set_title("Профілі середніх значень для кожного кластера [cite: 107]")
    ax_profile.set_ylabel("Значення показників")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_profile)

    st.markdown("---")
    st.subheader("Порівняння та висновки")

    st.markdown("""
        **Порівняйте результати ієрархічної кластеризації та методу k-середніх.**
        """)

    from scipy.cluster.hierarchy import fcluster
    clusters_hier = fcluster(Z1, k_choice, criterion='maxclust')
    df_cluster['Кластер (Ієрархічний)'] = clusters_hier - 1

    st.dataframe(df_cluster.iloc[:, [0, -2, -1]])

    st.text_area("**1. Які характеристики об'єктів у кожному кластері (за графіком профілів)?**",
                 placeholder="Наприклад: Кластер 0 має найвищий ВВП та тривалість життя. Кластер 1 характеризується високим рівнем безробіття...")

    st.text_area("**2. Чи узгоджуються результати двох процедур?**",
                 placeholder="Наприклад: Результати більшою мірою збігаються. Країни 'А' та 'Б' в обох методах потрапили до однієї групи, проте об'єкт 'В' за ієрархічним методом віднесено до іншої групи через...")
    st.subheader("Загальний висновок:")
    st.markdown("""
        - Метод k-середніх чутливий до вибору початкових центрів, але дає чіткіший поділ на великих даних[cite: 91].
        - Ієрархічний метод дозволяє візуально побачити 'спорідненість' об'єктів через дендрограму[cite: 27].
        - Узгодженість результатів двох методів свідчить про **достовірність** виділеної структури даних[cite: 19, 122].
        """)
