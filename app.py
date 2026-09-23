import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "accident_prediction_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
DEFAULT_DATA_PATH = "data/traffic_accidents.csv"
TARGET_COL = "crash_type"
DROP_COLS = ["crash_type", "crash_date"]

st.set_page_config(
    page_title="Traffic Accident Crash Type Predictor",
    page_icon="🚦",
    layout="wide",
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def models_exist() -> bool:
    return all(os.path.exists(p) for p in [MODEL_PATH, FEATURES_PATH, ENCODERS_PATH])


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    return model, feature_names, label_encoders


@st.cache_data(show_spinner=False)
def load_dataset(path_or_buffer):
    return pd.read_csv(path_or_buffer)


def prepare_features(df: pd.DataFrame, feature_names, label_encoders) -> pd.DataFrame:
    """Apply the same preprocessing used during training."""
    df = df.copy()

    # Fill missing values
    for col in df.select_dtypes(include="object").columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN")

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns with saved encoders
    for col in df.select_dtypes(include="object").columns:
        if col in label_encoders:
            le = label_encoders[col]
            classes = set(le.classes_)
            df[col] = df[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in classes else -1
            )

    # Drop non-feature columns
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    # Keep only training features and correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]


def train_and_save(df: pd.DataFrame, model_dir: str = MODEL_DIR):
    """Train a RandomForest model on the provided dataframe and save it."""
    os.makedirs(model_dir, exist_ok=True)

    df = df.dropna(subset=[TARGET_COL]).copy()

    # Fill missing
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop([c for c in DROP_COLS if c in df.columns], axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    joblib.dump(model, os.path.join(model_dir, "accident_prediction_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(model_dir, "feature_names.pkl"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))

    return model, label_encoders, list(X.columns), acc, X_test, y_test


# ------------------------------------------------------------------
# Sidebar / header
# ------------------------------------------------------------------
st.title("🚦 Traffic Accident Crash Type Predictor")
st.caption("Predict whether an accident results in injury/tow or is a drive-away, using a Random Forest classifier.")

tabs = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction (CSV)", "🛠️ Train / Retrain Model"])

# ------------------------------------------------------------------
# Tab 1 — Single Prediction
# ------------------------------------------------------------------
with tabs[0]:
    st.subheader("Enter accident details")

    if not models_exist():
        st.warning(
            "No trained model found in `models/`. "
            "Go to the **Train / Retrain Model** tab and train the model first."
        )
    else:
        model, feature_names, label_encoders = load_artifacts()

        # Try to load a sample dataset to know categorical options
        sample_df = None
        if os.path.exists(DEFAULT_DATA_PATH):
            try:
                sample_df = load_dataset(DEFAULT_DATA_PATH)
            except Exception:
                sample_df = None

        with st.form("single_prediction_form"):
            st.markdown("#### Input features")
            cols = st.columns(3)
            user_input = {}

            for i, feature in enumerate(feature_names):
                col = cols[i % 3]

                # Choose widget type based on encoder / dtype
                if feature in label_encoders:
                    le = label_encoders[feature]
                    options = [str(c) for c in le.classes_]
                    default_idx = 0
                    if sample_df is not None and feature in sample_df.columns:
                        mode_val = str(sample_df[feature].mode().iloc[0])
                        if mode_val in options:
                            default_idx = options.index(mode_val)

                    user_input[feature] = col.selectbox(
                        feature, options, index=default_idx, key=f"sel_{feature}"
                    )
                else:
                    default_val = 0.0
                    if sample_df is not None and feature in sample_df.columns:
                        try:
                            default_val = float(sample_df[feature].median())
                        except Exception:
                            default_val = 0.0

                    user_input[feature] = col.number_input(
                        feature, value=default_val, key=f"num_{feature}"
                    )

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([user_input])

            # Encode using saved encoders (numeric features pass through)
            for col in input_df.columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    val = str(input_df.at[0, col])
                    if val in le.classes_:
                        input_df.at[0, col] = le.transform([val])[0]
                    else:
                        input_df.at[0, col] = -1

            input_df = input_df[feature_names]

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            target_le = label_encoders[TARGET_COL]
            pred_label = target_le.inverse_transform([pred])[0]

            st.success(f"### Predicted crash type: **{pred_label}**")

            st.markdown("#### Class probabilities")
            proba_df = pd.DataFrame(
                {
                    "Class": target_le.classes_,
                    "Probability": proba,
                }
            ).sort_values("Probability", ascending=False)

            st.dataframe(
                proba_df.style.format({"Probability": "{:.4f}"}),
                use_container_width=True,
            )
            st.bar_chart(proba_df.set_index("Class"))


# ------------------------------------------------------------------
# Tab 2 — Batch Prediction
# ------------------------------------------------------------------
with tabs[1]:
    st.subheader("Upload a CSV file of accidents")

    if not models_exist():
        st.warning("No trained model found. Please train the model first.")
    else:
        model, feature_names, label_encoders = load_artifacts()

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                batch_df = None

            if batch_df is not None:
                st.markdown("**Preview of uploaded data**")
                st.dataframe(batch_df.head(10), use_container_width=True)

                original = batch_df.copy()

                try:
                    X_batch = prepare_features(batch_df, feature_names, label_encoders)
                    preds = model.predict(X_batch)
                    probas = model.predict_proba(X_batch)

                    target_le = label_encoders[TARGET_COL]
                    pred_labels = target_le.inverse_transform(preds)

                    result = original.copy()
                    result["predicted_crash_type"] = pred_labels
                    result["confidence"] = probas.max(axis=1)

                    st.success(f"Predicted {len(result)} rows.")
                    st.dataframe(result.head(50), use_container_width=True)

                    csv_bytes = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download predictions as CSV",
                        data=csv_bytes,
                        file_name="accident_predictions.csv",
                        mime="text/csv",
                    )

                    if TARGET_COL in original.columns:
                        st.markdown("#### Evaluation on uploaded data (target present)")
                        y_true_raw = original[TARGET_COL].astype(str)
                        valid_mask = y_true_raw.isin(target_le.classes_)
                        if valid_mask.any():
                            y_true = target_le.transform(y_true_raw[valid_mask])
                            y_pred = preds[valid_mask]
                            acc = accuracy_score(y_true, y_pred)
                            st.metric("Accuracy on uploaded data", f"{acc:.4f}")
                            st.text("Classification report:")
                            st.text(
                                classification_report(
                                    target_le.inverse_transform(y_true),
                                    target_le.inverse_transform(y_pred),
                                    zero_division=0,
                                )
                            )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ------------------------------------------------------------------
# Tab 3 — Train / Retrain
# ------------------------------------------------------------------
with tabs[2]:
    st.subheader("Train or retrain the model")

    st.markdown(
        "Upload a training CSV, or use the default file at "
        f"`{DEFAULT_DATA_PATH}` if it exists."
    )

    uploaded_train = st.file_uploader(
        "Upload training CSV", type=["csv"], key="train_uploader"
    )

    use_default = st.checkbox(
        f"Use default dataset at `{DEFAULT_DATA_PATH}`",
        value=os.path.exists(DEFAULT_DATA_PATH),
        disabled=not os.path.exists(DEFAULT_DATA_PATH),
    )

    if st.button("🚀 Train model"):
        train_df = None

        if uploaded_train is not None:
            try:
                train_df = pd.read_csv(uploaded_train)
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
        elif use_default and os.path.exists(DEFAULT_DATA_PATH):
            train_df = load_dataset(DEFAULT_DATA_PATH)
        else:
            st.error("Please upload a training CSV or enable the default dataset.")

        if train_df is not None:
            if TARGET_COL not in train_df.columns:
                st.error(f"Training CSV must contain a `{TARGET_COL}` column.")
            else:
                with st.spinner("Training model..."):
                    try:
                        _, _, _, acc, _, _ = train_and_save(train_df)
                        st.success(f"Model trained and saved. Test accuracy: **{acc:.4f}**")
                        st.cache_resource.clear()
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

    st.divider()
    st.markdown("#### Model status")

    if models_exist():
        st.success("✅ A trained model is available in `models/`.")
        try:
            mtime = os.path.getmtime(MODEL_PATH)
            st.caption(f"Last trained: {pd.to_datetime(mtime, unit='s')}")
        except Exception:
            pass
    else:
        st.error("❌ No trained model found. Please train one.")