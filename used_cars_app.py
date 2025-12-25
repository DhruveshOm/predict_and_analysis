import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ========================
# BASIC SETTINGS
# ========================
st.set_page_config(page_title="Used Cars App", layout="wide")
sns.set_style("whitegrid")


# ========================
# GLOBAL STYLE – PASTEL, CLEAN
# ========================

def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #f9fafb;
            --bg-card1: #fdf2ff;
            --bg-card2: #eff6ff;
            --border-soft: #e5e7eb;
            --text-main: #111827;
            --text-muted: #6b7280;
        }

        body, .main {
            background: radial-gradient(circle at top left, #fdf2ff, #eff6ff, #ecfeff);
            color: var(--text-main);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                         "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        .stButton>button {
            border-radius: 999px;
            padding: 0.6rem 1.6rem;
            border: 1px solid #fecdd3;
            background: linear-gradient(135deg,#f9a8d4,#c4b5fd);
            color: #312e81;
            font-weight: 600;
            letter-spacing: 0.02em;
            box-shadow: 0 6px 18px rgba(148,163,184,0.35);
        }
        .stButton>button:hover {
            filter: brightness(1.05);
            border-color: #a5b4fc;
            box-shadow: 0 8px 22px rgba(148,163,184,0.55);
        }

        .card {
            border-radius: 18px;
            padding: 1rem 1.3rem;
            background: linear-gradient(135deg, var(--bg-card1), var(--bg-card2));
            border: 1px solid var(--border-soft);
            box-shadow: 0 10px 32px rgba(148,163,184,0.25);
        }

        .card h3 {
            color: #1e293b;
            margin-bottom: 0.3rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.18rem 0.75rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            background: linear-gradient(135deg,#fee2e2,#e0f2fe);
            color: #4b5563;
            border: 1px solid #fecaca;
        }

        h1.hero {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            background: linear-gradient(120deg,#f97373,#fb7185,#f9a8d4,#a5b4fc);
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: 0.4rem;
        }

        p.subtitle {
            font-size: 0.95rem;
            color: var(--text-muted);
            max-width: 620px;
        }

        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #4b5563;
        }

        .dataframe td, .dataframe th {
            font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ========================
# HELPERS + DATA
# ========================

def categorize_price(price):
    if price <= 10000:
        return 'budget'
    elif price <= 30000:
        return 'midrange'
    else:
        return 'premium'


@st.cache_data
def preprocess_raw_csv():
    cols = [
        'price', 'year', 'manufacturer', 'model', 'condition', 'cylinders',
        'fuel', 'odometer', 'transmission', 'drive', 'type', 'state'
    ]
    chunk_size = 50_000
    df_list = []

    for chunk in pd.read_csv("vehicles.csv", usecols=cols, chunksize=chunk_size):
        chunk = chunk.dropna(subset=['price', 'year'])
        chunk = chunk[chunk['price'] > 100]
        chunk = chunk[chunk['price'] < 200000]
        chunk = chunk[chunk['year'] >= 1990]
        df_list.append(chunk)

    df = pd.concat(df_list, ignore_index=True)

    cat_cols = [
        'manufacturer', 'model', 'condition', 'cylinders', 'fuel',
        'transmission', 'drive', 'type', 'state'
    ]
    for col in cat_cols:
        df[col] = df[col].fillna('unknown')

    df['odometer'] = df['odometer'].fillna(df['odometer'].median())
    df.to_csv("used_cars_clean.csv", index=False)
    return df


@st.cache_data
def load_clean_data():
    return pd.read_csv("used_cars_clean.csv")


# ========================
# TRAINING FUNCTIONS
# ========================

@st.cache_resource
def train_regression_model():
    df = load_clean_data().copy()
    df['price_category'] = df['price'].apply(categorize_price)
    if 'model' in df.columns:
        df = df.drop(columns=['model'])

    X_reg = df.drop(columns=['price', 'price_category'])
    y_reg = df['price']

    sample_size = 20000
    if len(X_reg) > sample_size:
        X_reg = X_reg.sample(n=sample_size, random_state=42)
        y_reg = y_reg.loc[X_reg.index]

    cat_cols = X_reg.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_reg.select_dtypes(exclude=['object']).columns.tolist()

    X_reg_encoded = pd.get_dummies(X_reg, columns=cat_cols, drop_first=True)
    scaler = StandardScaler()
    X_reg_encoded[num_cols] = scaler.fit_transform(X_reg_encoded[num_cols])

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_encoded, y_reg, test_size=0.2, random_state=42
    )

    rf_reg = RandomForestRegressor(
        n_estimators=60, max_depth=18, random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = rf_reg.predict(X_test_reg)

    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)

    metrics = (mae, mse, rmse, r2)
    return rf_reg, scaler, X_reg_encoded.columns, num_cols, y_test_reg, y_pred_reg, metrics


@st.cache_resource
def train_classification_models():
    """
    Train classifiers and also return best model + preprocessing
    so we can do interactive predictions.
    """
    df = load_clean_data().copy()
    df['price_category'] = df['price'].apply(categorize_price)
    if 'model' in df.columns:
        df = df.drop(columns=['model'])

    X = df.drop(columns=['price', 'price_category'])
    y = df['price_category']

    sample_size = 40000
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)
        y = y.loc[X.index]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    scaler = StandardScaler()
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=18, random_state=42, n_jobs=-1
        ),
    }

    results, confusion_mats = [], {}
    class_names = ['budget', 'midrange', 'premium']
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append([name, acc, prec, rec, f1])
        confusion_mats[name] = confusion_matrix(y_test, y_pred, labels=class_names)
        fitted_models[name] = model

    results_df = pd.DataFrame(
        results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    ).sort_values(by="F1 Score", ascending=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    # Optional: feature importances for RF (if present)
    rf_importances = None
    if "Random Forest" in fitted_models:
        rf_model = fitted_models["Random Forest"]
        if hasattr(rf_model, "feature_importances_"):
            rf_importances = (rf_model.feature_importances_, X_encoded.columns)

    return (
        results_df,
        confusion_mats,
        class_names,
        best_model,
        scaler,
        X_encoded.columns,
        num_cols,
        rf_importances,
    )


# ========================
# PAGE HELPERS
# ========================

def back_to_home_button(key: str):
    """
    Back button – works (sets page to home).
    """
    left, mid, right = st.columns([1.2, 3, 1])
    with left:
        if st.button("⬅ Back to Home", key=key):
            st.session_state.page = "home"


# ========================
# PAGES
# ========================

def page_home():
    st.markdown(
        """
        <div class="card" style="margin-bottom: 1.3rem;">
          <span class="badge">used cars · ml app</span>
          <h1 class="hero">Used Cars Price Analytics</h1>
          <p class="subtitle">
            Clean the dataset, explore patterns, predict prices, and discover segments —
            all in one minimal pastel dashboard.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick stats (if dataset exists)
    try:
        df = load_clean_data()
        total = len(df)
        n_manuf = df['manufacturer'].nunique()
        median_price = df['price'].median()
        top_fuel = df['fuel'].value_counts().idxmax()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown('<div class="metric-label">Total Cars</div>', unsafe_allow_html=True)
        c1.markdown(f'<div class="metric-value">{total:,}</div>', unsafe_allow_html=True)

        c2.markdown('<div class="metric-label">Manufacturers</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-value">{n_manuf}</div>', unsafe_allow_html=True)

        c3.markdown('<div class="metric-label">Median Price</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-value">${median_price:,.0f}</div>', unsafe_allow_html=True)

        c4.markdown('<div class="metric-label">Most Common Fuel</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-value">{top_fuel}</div>', unsafe_allow_html=True)

        st.caption(
            f"Most cars in this dataset are **{df['price'].apply(categorize_price).mode()[0]}** "
            f"and typically use **{top_fuel}**."
        )
    except FileNotFoundError:
        st.info("Run **Preprocess Data** first to see dataset statistics here.")

    st.markdown("#### Choose a module")

    b1, b2, b3 = st.columns(3)
    b4, b5, b6 = st.columns(3)

    with b1:
        if st.button("🧹 Preprocess Data", use_container_width=True):
            st.session_state.page = "preprocess"
    with b2:
        if st.button("📊 EDA", use_container_width=True):
            st.session_state.page = "eda"
    with b3:
        if st.button("🎯 Regression", use_container_width=True):
            st.session_state.page = "regression"

    with b4:
        if st.button("🧮 Classification", use_container_width=True):
            st.session_state.page = "classification"
    with b5:
        if st.button("📌 PCA + K-Means", use_container_width=True):
            st.session_state.page = "clustering"
    with b6:
        if st.button("🌳 Hierarchical", use_container_width=True):
            st.session_state.page = "hierarchical"


def page_preprocess():
    st.markdown(
        """
        <div class="card">
          <span class="badge">step 1</span>
          <h3>🧹 Preprocess Raw Dataset</h3>
          <p class="subtitle">
            Load <code>vehicles.csv</code>, remove junk values and fill missing data.
            Output is <code>used_cars_clean.csv</code>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Run Preprocessing ✨"):
        with st.spinner("Processing raw CSV..."):
            df = preprocess_raw_csv()
        st.success("Done! Saved as `used_cars_clean.csv`.")
        st.write("Shape:", df.shape)
        st.dataframe(df.head())

    st.markdown("---")
    back_to_home_button("back_preprocess")


def page_eda(df):
    st.markdown(
        """
        <div class="card">
          <span class="badge">step 2</span>
          <h3>📊 EDA Overview</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Interactive filters ----
    st.markdown("##### Filters")

    top_manufs = df['manufacturer'].value_counts().index[:10].tolist()
    manuf_options = ["All"] + sorted(top_manufs)
    c1, c2, c3 = st.columns(3)

    with c1:
        selected_manuf = st.selectbox("Manufacturer (top 10)", manuf_options)
    with c2:
        min_price = int(df['price'].min())
        max_price = int(df['price'].max())
        price_range = st.slider(
            "Price range",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=500
        )
    with c3:
        hide_outliers = st.checkbox("Hide extreme price outliers (IQR-based)")

    df_eda = df[
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1])
    ]
    if selected_manuf != "All":
        df_eda = df_eda[df_eda['manufacturer'] == selected_manuf]

    if hide_outliers and len(df_eda) > 0:
        q1, q3 = df_eda['price'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_eda = df_eda[(df_eda['price'] >= lower) & (df_eda['price'] <= upper)]

    st.write("Filtered shape:", df_eda.shape)

    if len(df_eda) > 0:
        avg_price = df_eda['price'].mean()
        avg_year = df_eda['year'].mean()
        avg_odo = df_eda['odometer'].mean()
        st.caption(
            f"In the selected filter, average price is **${avg_price:,.0f}**, "
            f"average year is **{avg_year:.0f}**, and average mileage is "
            f"**{avg_odo:,.0f} km**."
        )

    st.markdown("---")
    st.subheader("Boxplots: Price · Year · Odometer")

    numeric_cols = ['price', 'year', 'odometer']
    if len(df_eda) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for i, col in enumerate(numeric_cols):
            sns.boxplot(x=df_eda[col], color='#bfdbfe', ax=axes[i])
            axes[i].set_title(col.capitalize())
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No data to show for current filter.")

    st.markdown("---")
    st.subheader("Scatter Plots (Filtered)")

    if len(df_eda) > 0:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        sns.scatterplot(data=df_eda, x='odometer', y='price', alpha=0.3, ax=axes2[0])
        axes2[0].set_title("Price vs Odometer")
        sns.scatterplot(data=df_eda, x='year', y='price', alpha=0.3, ax=axes2[1])
        axes2[1].set_title("Price vs Year")
        plt.tight_layout()
        st.pyplot(fig2)

    # Manufacturer comparison
    st.markdown("---")
    st.subheader("Compare Manufacturers (Price Distribution)")

    all_manufs = df['manufacturer'].value_counts().index.tolist()
    selected_many = st.multiselect(
        "Select up to 3 manufacturers",
        options=all_manufs,
        default=all_manufs[:3]
    )
    selected_many = selected_many[:3]

    if selected_many:
        df_comp = df[df['manufacturer'].isin(selected_many)]
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df_comp,
            x='manufacturer',
            y='price',
            ax=ax3
        )
        ax3.set_title("Price distribution by manufacturer")
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("---")
    back_to_home_button("back_eda")


def page_regression(df):
    st.markdown(
        """
        <div class="card">
          <span class="badge">step 3</span>
          <h3>🎯 Price Regression (Random Forest)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading / training model (cached)..."):
        rf_reg, scaler, feature_cols, num_cols, y_test_reg, y_pred_reg, metrics = train_regression_model()

    mae, mse, rmse, r2 = metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-label">MAE</div>', unsafe_allow_html=True)
    c1.markdown(f'<div class="metric-value">{mae:.0f}</div>', unsafe_allow_html=True)

    c2.markdown('<div class="metric-label">RMSE</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-value">{rmse:.0f}</div>', unsafe_allow_html=True)

    c3.markdown('<div class="metric-label">R²</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-value">{r2:.3f}</div>', unsafe_allow_html=True)

    c4.markdown('<div class="metric-label">Test Samples</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-value">{len(y_test_reg):,}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Predicted vs Actual")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test_reg, y_pred_reg, alpha=0.4)
    ax.plot(
        [y_test_reg.min(), y_test_reg.max()],
        [y_test_reg.min(), y_test_reg.max()],
        'k--', linewidth=2
    )
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔮 Try Your Own Inputs (Price Prediction)")

    df_local = df.copy()
    df_local['price_category'] = df_local['price'].apply(categorize_price)
    if 'model' in df_local.columns:
        df_local = df_local.drop(columns=['model'])

    manufacturers = sorted(df_local['manufacturer'].unique().tolist())
    conditions = sorted(df_local['condition'].unique().tolist())
    cylinders = sorted(df_local['cylinders'].unique().tolist())
    fuels = sorted(df_local['fuel'].unique().tolist())
    transmissions = sorted(df_local['transmission'].unique().tolist())
    drives = sorted(df_local['drive'].unique().tolist())
    types_ = sorted(df_local['type'].unique().tolist())
    states = sorted(df_local['state'].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
        odometer = st.number_input("Odometer (km)", min_value=0, max_value=500000, value=60000)
        manufacturer = st.selectbox("Manufacturer", manufacturers)
        condition = st.selectbox("Condition", conditions)
        fuel = st.selectbox("Fuel", fuels)
    with col2:
        cyl = st.selectbox("Cylinders", cylinders)
        transmission = st.selectbox("Transmission", transmissions)
        drive = st.selectbox("Drive", drives)
        car_type = st.selectbox("Type", types_)
        state = st.selectbox("State", states)

    if st.button("Predict Price 💸"):
        user = {
            'year': [year],
            'manufacturer': [manufacturer],
            'condition': [condition],
            'cylinders': [cyl],
            'fuel': [fuel],
            'odometer': [odometer],
            'transmission': [transmission],
            'drive': [drive],
            'type': [car_type],
            'state': [state]
        }
        user_df = pd.DataFrame(user)
        cat_cols_user = user_df.select_dtypes(include=['object']).columns.tolist()
        user_encoded = pd.get_dummies(user_df, columns=cat_cols_user, drop_first=True)
        user_encoded = user_encoded.reindex(columns=feature_cols, fill_value=0)
        user_encoded[num_cols] = scaler.transform(user_encoded[num_cols])

        pred_price = rf_reg.predict(user_encoded)[0]

        # Confidence band based on RMSE
        lower = max(pred_price - rmse, 0)
        upper = pred_price + rmse

        st.success(f"Estimated Price: **${pred_price:,.2f}**")
        st.caption(f"Typical error (±RMSE): price likely in **${lower:,.0f} – ${upper:,.0f}**.")

        # --- Confidence score (how "correct-ish" this is likely to be) ---

        # 1) Density: how many similar cars in dataset? (same manufacturer + type)
        df_sim_all = df_local[
            (df_local['manufacturer'] == manufacturer) &
            (df_local['type'] == car_type)
        ]
        sim_count = len(df_sim_all)
        conf_density = min(sim_count / 1000, 1.0)  # normalize: 1000+ rows = 1.0

        # 2) Model quality from R² (map roughly from 0.5–1.0 → 0–1)
        conf_model = (r2 - 0.5) / 0.5
        conf_model = max(0.0, min(conf_model, 1.0))

        # 3) Combine
        confidence = 0.6 * conf_model + 0.4 * conf_density
        confidence_pct = int(confidence * 100)

        st.markdown("**Model confidence for this prediction:**")
        st.progress(confidence_pct)

        if confidence >= 0.8:
            st.success(f"High confidence (**{confidence_pct}%**): model has seen many similar cars and fits well.")
        elif confidence >= 0.6:
            st.info(f"Moderate confidence (**{confidence_pct}%**): prediction is reasonable but not perfect.")
        else:
            st.warning(
                f"Low confidence (**{confidence_pct}%**): this combination is rare or far from training data, "
                f"so treat this price as a rough estimate."
            )

        # Similar cars from dataset (top 5 nearest)
        if sim_count > 0:
            df_sim = df_sim_all.assign(
                dist=np.abs(df_sim_all['year'] - year) + np.abs(df_sim_all['odometer'] - odometer) / 10000
            )
            df_sim = df_sim.sort_values('dist').head(5)
            st.markdown("**Closest 5 similar cars in dataset:**")
            st.dataframe(df_sim[['price', 'year', 'manufacturer', 'condition',
                                 'fuel', 'odometer', 'transmission', 'drive', 'type']])
        else:
            st.info("No very similar cars found in this filtered dataset.")

    st.markdown("---")
    back_to_home_button("back_regression")


def page_classification(df):
    st.markdown(
        """
        <div class="card">
          <span class="badge">step 4</span>
          <h3>🧮 Price Category Classification</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading / training models (cached)..."):
        (
            results_df,
            confusion_mats,
            class_names,
            best_model,
            clf_scaler,
            clf_feature_cols,
            clf_num_cols,
            rf_importances,
        ) = train_classification_models()

    st.subheader("Model Metrics")
    st.dataframe(results_df.set_index("Model"))

    st.subheader("F1 Score Comparison")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=results_df, x="Model", y="F1 Score", ax=ax, palette="pastel")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("F1 Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # Confusion matrix selector
    st.subheader("Confusion Matrix Viewer")
    selected_model_for_cm = st.selectbox(
        "Select model",
        results_df["Model"].tolist()
    )
    cm = confusion_mats[selected_model_for_cm]

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="BuPu",
        xticklabels=class_names, yticklabels=class_names, ax=ax_cm
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig_cm)

    # Feature importance chart for Random Forest
    if rf_importances is not None:
        importances, feat_cols = rf_importances
        imp_series = pd.Series(importances, index=feat_cols)
        top_imp = imp_series.sort_values(ascending=False).head(8)

        st.subheader("Top 8 Important Features (Random Forest)")
        fig_imp, ax_imp = plt.subplots(figsize=(7, 4))
        sns.barplot(x=top_imp.values, y=top_imp.index, ax=ax_imp)
        ax_imp.set_xlabel("Importance")
        ax_imp.set_ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig_imp)

    # ---- Interactive classification prediction ----
    st.markdown("---")
    st.subheader("🔮 Try Your Own Inputs (Category Prediction)")

    df_local = df.copy()
    df_local['price_category'] = df_local['price'].apply(categorize_price)
    if 'model' in df_local.columns:
        df_local = df_local.drop(columns=['model'])

    manufacturers = sorted(df_local['manufacturer'].unique().tolist())
    conditions = sorted(df_local['condition'].unique().tolist())
    cylinders = sorted(df_local['cylinders'].unique().tolist())
    fuels = sorted(df_local['fuel'].unique().tolist())
    transmissions = sorted(df_local['transmission'].unique().tolist())
    drives = sorted(df_local['drive'].unique().tolist())
    types_ = sorted(df_local['type'].unique().tolist())
    states = sorted(df_local['state'].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year ", min_value=1990, max_value=2025, value=2015, key="clf_year")
        odometer = st.number_input("Odometer (km) ", min_value=0, max_value=500000,
                                   value=60000, key="clf_odo")
        manufacturer = st.selectbox("Manufacturer ", manufacturers, key="clf_manuf")
        condition = st.selectbox("Condition ", conditions, key="clf_cond")
        fuel = st.selectbox("Fuel ", fuels, key="clf_fuel")
    with col2:
        cyl = st.selectbox("Cylinders ", cylinders, key="clf_cyl")
        transmission = st.selectbox("Transmission ", transmissions, key="clf_trans")
        drive = st.selectbox("Drive ", drives, key="clf_drive")
        car_type = st.selectbox("Type ", types_, key="clf_type")
        state = st.selectbox("State ", states, key="clf_state")

    if st.button("Predict Category 🏷️"):
        user = {
            'year': [year],
            'manufacturer': [manufacturer],
            'condition': [condition],
            'cylinders': [cyl],
            'fuel': [fuel],
            'odometer': [odometer],
            'transmission': [transmission],
            'drive': [drive],
            'type': [car_type],
            'state': [state]
        }
        user_df = pd.DataFrame(user)
        cat_cols_user = user_df.select_dtypes(include=['object']).columns.tolist()
        user_encoded = pd.get_dummies(user_df, columns=cat_cols_user, drop_first=True)
        user_encoded = user_encoded.reindex(columns=clf_feature_cols, fill_value=0)
        user_encoded[clf_num_cols] = clf_scaler.transform(user_encoded[clf_num_cols])

        pred_class = best_model.predict(user_encoded)[0]
        st.success(f"Predicted Price Category: **{pred_class.upper()}**")

        # Show probabilities as bar chart (if available) + confidence label
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(user_encoded)[0]
            prob_df = pd.DataFrame({
                "Category": best_model.classes_,
                "Probability": probs
            })
            prob_df = prob_df.sort_values("Probability", ascending=False)

            st.caption("Model confidence for each category:")
            fig_prob, ax_prob = plt.subplots(figsize=(5, 3))
            sns.barplot(data=prob_df, x="Category", y="Probability", ax=ax_prob)
            ax_prob.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig_prob)

            max_prob = float(prob_df["Probability"].max())
            conf_pct = int(max_prob * 100)
            st.markdown("**Prediction confidence (based on probability):**")
            st.progress(conf_pct)

            if max_prob >= 0.8:
                st.success(f"High confidence (**{conf_pct}%**): model is quite sure about this category.")
            elif max_prob >= 0.6:
                st.info(f"Moderate confidence (**{conf_pct}%**): prediction is reasonable but borderline.")
            else:
                st.warning(
                    f"Low confidence (**{conf_pct}%**): model is unsure, so treat this label with caution."
                )

    # Simple "what-if" odometer slider – show predicted class for same car but variable mileage
    st.markdown("---")
    st.subheader("⚙️ What-if Analysis: Change Odometer Only")

    whatif_odo = st.slider(
        "Simulated odometer (km)",
        min_value=0,
        max_value=500000,
        value=60000,
        step=10000
    )

    # Using same last selected feature values
    base_user = {
        'year': [year],
        'manufacturer': [manufacturer],
        'condition': [condition],
        'cylinders': [cyl],
        'fuel': [fuel],
        'odometer': [whatif_odo],
        'transmission': [transmission],
        'drive': [drive],
        'type': [car_type],
        'state': [state]
    }
    base_df = pd.DataFrame(base_user)
    base_cat_cols = base_df.select_dtypes(include=['object']).columns.tolist()
    base_encoded = pd.get_dummies(base_df, columns=base_cat_cols, drop_first=True)
    base_encoded = base_encoded.reindex(columns=clf_feature_cols, fill_value=0)
    base_encoded[clf_num_cols] = clf_scaler.transform(base_encoded[clf_num_cols])

    whatif_class = best_model.predict(base_encoded)[0]
    st.caption(
        f"For the same car but with **{whatif_odo:,} km** driven, "
        f"predicted category becomes **{whatif_class.upper()}**."
    )

    st.markdown("---")
    back_to_home_button("back_classification")


def page_clustering(df):
    st.markdown(
        """
        <div class="card">
          <span class="badge">step 5</span>
          <h3>📌 PCA + K-Means Clusters</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = df.copy()
    df['price_category'] = df['price'].apply(categorize_price)
    if 'model' in df.columns:
        df = df.drop(columns=['model'])

    sample_size = 8000
    df_sample = df.sample(n=sample_size, random_state=42)
    X_sample = df_sample.drop(columns=['price', 'price_category'])
    X_sample_encoded = pd.get_dummies(X_sample, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample_encoded)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Elbow method (fixed)
    inertia_list, K_range = [], range(2, 6)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_pca)
        inertia_list.append(kmeans.inertia_)

    st.subheader("Elbow Method")
    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
    ax_elbow.plot(list(K_range), inertia_list, marker='o')
    ax_elbow.set_xlabel("K")
    ax_elbow.set_ylabel("Inertia")
    plt.tight_layout()
    st.pyplot(fig_elbow)

    st.markdown("---")
    # Interactive K + color mode
    c1, c2 = st.columns(2)
    with c1:
        k_val = st.slider("Choose K for visualization", min_value=2, max_value=6, value=3)
    with c2:
        color_by = st.radio(
            "Color points by",
            ["Cluster", "Price Category"],
            horizontal=True
        )

    kmeans_final = KMeans(n_clusters=k_val, random_state=42)
    clusters = kmeans_final.fit_predict(X_pca)
    df_sample['cluster'] = clusters

    sil_score_val = silhouette_score(X_pca, clusters)
    st.write(f"Silhouette Score (K={k_val}): **{sil_score_val:.4f}**")

    st.subheader("Clusters on PCA(2D)")
    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
    palette = ['#f9a8d4', '#a5b4fc', '#99f6e4', '#fde68a', '#bfdbfe']

    if color_by == "Cluster":
        hue_data = df_sample['cluster']
        pal = palette[:k_val]
        legend_title = "Cluster"
    else:
        hue_data = df_sample['price_category']
        pal = ['#a5b4fc', '#99f6e4', '#f9a8d4']
        legend_title = "Price Category"

    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=hue_data,
        palette=pal,
        s=18, ax=ax_scatter
    )
    ax_scatter.set_xlabel("PC 1")
    ax_scatter.set_ylabel("PC 2")
    ax_scatter.legend(title=legend_title)
    plt.tight_layout()
    st.pyplot(fig_scatter)

    # Cluster summaries
    st.subheader("Cluster Summary (by KMeans clusters)")
    cluster_summary = df_sample.groupby('cluster').agg({
        'price': 'mean',
        'year': 'mean',
        'odometer': 'mean',
        'manufacturer': lambda x: x.value_counts().idxmax()
    })
    cluster_summary = cluster_summary.rename(columns={
        'price': 'avg_price',
        'year': 'avg_year',
        'odometer': 'avg_odometer',
        'manufacturer': 'top_manufacturer'
    })
    st.dataframe(cluster_summary.style.format({
        'avg_price': '{:,.0f}',
        'avg_year': '{:,.0f}',
        'avg_odometer': '{:,.0f}'
    }))

    st.markdown("---")
    back_to_home_button("back_clustering")


def page_hierarchical(df):
    st.markdown(
        """
        <div class="card">
          <span class="badge">step 6</span>
          <h3>🌳 Hierarchical Dendrogram</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = df.copy()
    df['price_category'] = df['price'].apply(categorize_price)
    if 'model' in df.columns:
        df = df.drop(columns=['model'])

    # Interactive sample size
    sample_size = st.slider(
        "Sample size for dendrogram",
        min_value=100, max_value=500, value=300, step=50
    )

    df_hier = df.sample(n=sample_size, random_state=42)
    X_hier = df_hier.drop(columns=['price', 'price_category'], errors='ignore')
    X_hier_enc = pd.get_dummies(X_hier, drop_first=True)

    # Use at most 200 rows for linkage
    n_rows_linkage = min(200, len(X_hier_enc))
    data_for_linkage = X_hier_enc.iloc[:n_rows_linkage]

    linkage_matrix = linkage(data_for_linkage, method='ward')

    fig_dendro, ax_dendro = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=15,
        leaf_rotation=45,
        ax=ax_dendro
    )
    ax_dendro.set_xlabel("Cluster index / size")
    ax_dendro.set_ylabel("Distance")
    plt.tight_layout()
    st.pyplot(fig_dendro)

    # Cut distance slider + cluster count
    dist_min = float(linkage_matrix[:, 2].min())
    dist_max = float(linkage_matrix[:, 2].max())
    default_cut = float(np.median(linkage_matrix[:, 2]))

    cut_dist = st.slider(
        "Cut distance (to estimate number of clusters)",
        min_value=dist_min,
        max_value=dist_max,
        value=default_cut
    )

    cluster_labels = fcluster(linkage_matrix, t=cut_dist, criterion='distance')
    n_clusters = len(np.unique(cluster_labels))
    st.write(
        f"At cut distance **{cut_dist:.2f}**, the dendrogram would form "
        f"approximately **{n_clusters} clusters**."
    )

    st.markdown("---")
    back_to_home_button("back_hierarchical")


# ========================
# MAIN – BUTTON NAV ONLY
# ========================

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    page = st.session_state.page

    if page == "home":
        page_home()
    elif page == "preprocess":
        page_preprocess()
    else:
        try:
            df = load_clean_data()
        except FileNotFoundError:
            st.error("`used_cars_clean.csv` not found. Run **Preprocess Data** first.")
            back_to_home_button("back_missing")
            return

        if page == "eda":
            page_eda(df)
        elif page == "regression":
            page_regression(df)
        elif page == "classification":
            page_classification(df)
        elif page == "clustering":
            page_clustering(df)
        elif page == "hierarchical":
            page_hierarchical(df)


if __name__ == "__main__":
    main()

