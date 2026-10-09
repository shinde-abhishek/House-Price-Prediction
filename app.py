from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"

TARGET_COLUMN = "Price"

st.title("🏠 House Price Prediction")
st.caption("Estimate a house price using machine learning.")
st.divider()


# ============================================================
# FIND AND LOAD DATASET
# ============================================================
def find_dataset():
    """Find a CSV inside the project's data folder, then project root."""
    candidates = []
    for folder in (DATA_DIR, BASE_DIR):
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.csv")))
    # Prefer the user's known dataset filename.
    preferred = DATA_DIR / "House Price Prediction Dataset (1).csv"
    if preferred.exists():
        return preferred
    return candidates[0] if candidates else None


@st.cache_data(show_spinner=False)
def load_dataset(csv_path, modified_time):
    # modified_time is part of the cache key so file changes are reloaded.
    return pd.read_csv(csv_path)


dataset_path = find_dataset()
if dataset_path is None:
    st.error("Dataset not found. Put your CSV file inside the project's data folder.")
    st.code(
        "House_Price_Prediction/\n"
        "├── app.py\n"
        "├── data/\n"
        "│   └── House Price Prediction Dataset (1).csv\n"
        "└── models/"
    )
    st.stop()

try:
    df = load_dataset(str(dataset_path), dataset_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

# Offer target selection if target is not literally called "Price".
if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")
    st.write("Columns available in your CSV:", df.columns.tolist())
    st.info("Update TARGET_COLUMN near the top of app.py to the actual selling-price column.")
    st.stop()

# Remove rows without a target value; a target must exist for training.
df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Remove rows with no useful feature data and columns that are entirely empty.
df = df.dropna(axis=1, how="all")
if TARGET_COLUMN not in df.columns or df.empty:
    st.error("No usable target values remain after cleaning the dataset.")
    st.stop()

# Exclude obvious ID-like columns, if present.
X_all = df.drop(columns=[TARGET_COLUMN]).copy()
y_all = df[TARGET_COLUMN].copy()
id_like = [
    col for col in X_all.columns
    if col.strip().lower() in {"id", "index", "unnamed: 0"}
]
if id_like:
    X_all = X_all.drop(columns=id_like)

if X_all.shape[1] == 0:
    st.error("No feature columns are available for training.")
    st.stop()

# Drop rows where every feature is missing.
valid_rows = ~X_all.isna().all(axis=1)
X = X_all.loc[valid_rows].copy()
y = y_all.loc[valid_rows].copy()

if len(X) < 5:
    st.error("At least 5 usable rows are recommended to train and evaluate the model.")
    st.stop()

numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

# Convert boolean columns to numeric for a consistent pipeline.
for col in X.select_dtypes(include=["bool"]).columns:
    X[col] = X[col].astype(int)
    if col not in numeric_columns:
        numeric_columns.append(col)

# Remove constant columns; they add no predictive information.
constant_columns = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
if constant_columns:
    X = X.drop(columns=constant_columns)
    numeric_columns = [c for c in numeric_columns if c not in constant_columns]
    categorical_columns = [c for c in categorical_columns if c not in constant_columns]

if X.shape[1] == 0:
    st.error("All feature columns are constant or empty. Please check your dataset.")
    st.stop()

numeric_columns = [c for c in numeric_columns if c in X.columns]
categorical_columns = [c for c in categorical_columns if c in X.columns]

with st.expander("📊 Dataset overview", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows used", f"{len(X):,}")
    c2.metric("Input features", len(X.columns))
    c3.metric("Target column", TARGET_COLUMN)
    st.write("Dataset file:", str(dataset_path.relative_to(BASE_DIR)))
    st.write("Feature columns:", X.columns.tolist())
    st.dataframe(df.head(), use_container_width=True)


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================
def make_preprocessor():
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def model_candidates():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "KNN": KNeighborsRegressor(n_neighbors=min(5, max(1, len(X) - 1))),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "SVR": SVR(),
    }


@st.cache_resource(show_spinner=False)
def train_models(dataframe, target_series, data_signature):
    # Cache key includes a signature derived from the loaded data.
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe, target_series, test_size=0.2, random_state=42
    )

    trained = {}
    rows = []
    errors = []

    for name, estimator in model_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", estimator),
            ]
        )
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            mae = float(mean_absolute_error(y_test, predictions))
            r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else float("nan")

            trained[name] = pipeline
            rows.append({"Model": name, "R² Score": r2, "MAE": mae, "RMSE": rmse})
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if not rows:
        raise RuntimeError("All models failed. " + " | ".join(errors))

    scores = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    # Choose lowest RMSE on the held-out test set.
    winner_name = str(scores.iloc[0]["Model"])
    return trained, scores, winner_name, errors


data_signature = (
    str(dataset_path.stat().st_mtime_ns),
    len(X),
    tuple(X.columns),
    float(y.sum()),
)
with st.spinner("Training and comparing regression models. This may take a little while the first time..."):
    try:
        trained_models, results_df, best_model_name, model_errors = train_models(
            X, y, data_signature
        )
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

best_model = trained_models[best_model_name]

# Persist the selected model and metadata.
try:
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_columns": X.columns.tolist(),
            "target_column": TARGET_COLUMN,
            "dataset_path": str(dataset_path),
        },
        MODEL_PATH,
    )
except Exception as exc:
    st.warning(f"Model trained, but could not be saved: {exc}")

with st.expander("🏆 Model comparison", expanded=False):
    st.success(f"Best model selected by lowest test RMSE: {best_model_name}")
    st.dataframe(
        results_df.style.format(
            {"R² Score": "{:.4f}", "MAE": "{:,.2f}", "RMSE": "{:,.2f}"}
        ),
        use_container_width=True,
    )
    if model_errors:
        st.caption("Some models could not be trained:")
        for message in model_errors:
            st.write(f"- {message}")
    st.caption("Scores are from a single train/test split; they are estimates, not a guarantee of future accuracy.")


# ============================================================
# PREDICTION FORM
# ============================================================
st.subheader("🏡 Enter House Details")
st.write("Use the dataset's feature ranges. The model will estimate the selling price.")

with st.form("house_prediction_form"):
    input_data = {}
    columns = st.columns(2)

    for idx, column in enumerate(X.columns):
        with columns[idx % 2]:
            series = X[column]
            if column in categorical_columns:
                options = series.dropna().astype(str).unique().tolist()
                if not options:
                    options = ["Unknown"]
                input_data[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options=options,
                    key=f"input_{column}",
                )
            else:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                if numeric_series.empty:
                    default_value, minimum, maximum = 0.0, 0.0, 1.0
                else:
                    minimum = float(numeric_series.min())
                    maximum = float(numeric_series.max())
                    default_value = float(numeric_series.median())
                    if minimum == maximum:
                        # number_input requires max_value >= min_value; allow exact constant.
                        maximum = minimum
                input_data[column] = st.number_input(
                    column.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default_value, minimum), maximum),
                    key=f"input_{column}",
                )

    submitted = st.form_submit_button(
        "🔮 Predict House Price", use_container_width=True
    )

if submitted:
    try:
        input_df = pd.DataFrame([input_data], columns=X.columns)

        # Restore numeric dtypes for model input.
        for col in numeric_columns:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

        prediction = float(best_model.predict(input_df)[0])
        st.success(f"🏠 Estimated House Price: ₹{prediction:,.2f}")
        st.caption(
            "This is a machine-learning estimate based on the supplied dataset, "
            "not a formal property valuation."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(f"Model file: {MODEL_PATH.relative_to(BASE_DIR)}")
