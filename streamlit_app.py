import os
import io
import json
import base64
import hashlib

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ============================================================
# CONFIG
# ============================================================

PROJECT = os.environ.get(
    "CYBER_PROJECT_DIR",
    "/content/drive/MyDrive/AI_Cybersecurity_Project",
)

MODEL_DIR = os.path.join(PROJECT, "models")
RESULT_DIR = os.path.join(PROJECT, "results")
KEY_DIR = os.path.join(PROJECT, "keys")

PACKAGE_PATH = os.path.join(MODEL_DIR, "secure_model_package.json")
AES_KEY_PATH = os.path.join(KEY_DIR, "aes256_key_demo.bin")
PACKAGE_HASH_PATH = os.path.join(
    RESULT_DIR, "secure_package_sha256.txt"
)

MAX_UPLOAD_ROWS = 100_000


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI-Based Cyber Threat Detection")
st.caption(
    "CIC-IDS2017 • Extra Trees • Secure AI Model Deployment"
)

st.markdown(
    """
This dashboard uses the **cryptographically protected final model**.
Upload a CIC-IDS2017 flow CSV and the system classifies each flow as
**BENIGN** or **ATTACK** using the locked operating threshold.
"""
)


# ============================================================
# SECURE MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Verifying and loading secure AI model...")
def load_secure_model():
    # ---- Check required files ----
    required = [
        PACKAGE_PATH,
        AES_KEY_PATH,
        PACKAGE_HASH_PATH,
    ]

    missing = [p for p in required if not os.path.exists(p)]

    if missing:
        raise FileNotFoundError(
            "Missing project files:\n" + "\n".join(missing)
        )

    # ---- SHA-256 fingerprint of package ----
    with open(PACKAGE_PATH, "rb") as f:
        package_bytes = f.read()

    actual_hash = hashlib.sha256(package_bytes).hexdigest()

    with open(PACKAGE_HASH_PATH, "r") as f:
        expected_hash = f.read().strip()

    hash_verified = actual_hash == expected_hash

    if not hash_verified:
        raise ValueError(
            "Secure package SHA-256 fingerprint does not match."
        )

    # ---- Read package ----
    with open(PACKAGE_PATH, "r") as f:
        package = json.load(f)

    nonce = base64.b64decode(
        package["encryption"]["nonce"]
    )

    aad = base64.b64decode(
        package["encryption"]["aad"]
    )

    ciphertext = base64.b64decode(
        package["ciphertext"]
    )

    signature = base64.b64decode(
        package["signature"]["signature"]
    )

    public_key = Ed25519PublicKey.from_public_bytes(
        base64.b64decode(
            package["signature"]["public_key"]
        )
    )

    # ---- Ed25519 signature verification ----
    try:
        public_key.verify(
            signature,
            aad + nonce + ciphertext,
        )
        signature_verified = True
    except InvalidSignature as exc:
        raise ValueError(
            "Ed25519 signature verification failed."
        ) from exc

    # ---- AES-256-GCM decryption ----
    with open(AES_KEY_PATH, "rb") as f:
        aes_key = f.read()

    decrypted_bytes = AESGCM(aes_key).decrypt(
        nonce,
        ciphertext,
        aad,
    )

    payload = joblib.load(
        io.BytesIO(decrypted_bytes)
    )

    model = payload["model"]
    imputer = payload["imputer"]
    metadata = payload["metadata"]

    return {
        "model": model,
        "imputer": imputer,
        "metadata": metadata,
        "hash_verified": hash_verified,
        "signature_verified": signature_verified,
        "package_size": len(package_bytes),
    }


try:
    secure = load_secure_model()
except Exception as exc:
    st.error("Secure model could not be loaded.")
    st.exception(exc)
    st.stop()


model = secure["model"]
imputer = secure["imputer"]
metadata = secure["metadata"]

features = metadata["features"]
threshold = float(metadata["decision_threshold"])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🔐 Model Security")

    st.success("SHA-256 verified")
    st.success("Ed25519 signature verified")
    st.success("AES-256-GCM decryption verified")

    st.divider()

    st.subheader("Model")
    st.write(f"**Algorithm:** {type(model).__name__}")
    st.write(f"**Features:** {len(features)}")
    st.write(f"**Threshold:** {threshold:.3f}")

    st.divider()

    st.subheader("Evaluation")
    st.write("**Training:** Monday–Thursday")
    st.write("**Final test:** Friday")

    st.divider()

    st.caption(
        "This dashboard performs binary detection only: "
        "BENIGN vs ATTACK."
    )


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Model", "Extra Trees")

with c2:
    st.metric("Input Features", len(features))

with c3:
    st.metric("Decision Threshold", f"{threshold:.3f}")

with c4:
    st.metric("Model Package", f"{secure['package_size'] / 1024 / 1024:.1f} MB")


st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📁 Analyze Network Traffic")

uploaded_file = st.file_uploader(
    "Upload a CIC-IDS2017 flow CSV",
    type=["csv"],
    help=(
        f"For a responsive demo, the dashboard analyzes at most "
        f"{MAX_UPLOAD_ROWS:,} rows per upload."
    ),
)


if uploaded_file is None:
    st.info(
        "Upload a CIC-IDS2017 CSV to start detection."
    )

    with st.expander("What does the dashboard expect?"):
        st.write(
            """
            The CSV should contain the same CIC-IDS2017 traffic-flow
            features used during training. Columns such as `Label`,
            `SOURCE_FILE`, and `Destination Port` are not used by the
            final model.
            """
        )

    st.stop()


# ============================================================
# READ DATA
# ============================================================

try:
    df = pd.read_csv(
        uploaded_file,
        nrows=MAX_UPLOAD_ROWS,
    )
except Exception as exc:
    st.error("Could not read the uploaded CSV.")
    st.exception(exc)
    st.stop()


df.columns = df.columns.astype(str).str.strip()

if len(df) == MAX_UPLOAD_ROWS:
    st.warning(
        f"Demo limit reached: only the first {MAX_UPLOAD_ROWS:,} "
        "rows were analyzed."
    )


# ============================================================
# FEATURE CHECK
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:
    st.error(
        f"The uploaded file is missing {len(missing_features)} "
        "required model features."
    )

    st.write("Missing columns:")
    st.code("\n".join(missing_features[:30]))

    st.stop()


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[features].copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce",
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

try:
    X_imp = imputer.transform(X)
except Exception as exc:
    st.error("Feature preprocessing failed.")
    st.exception(exc)
    st.stop()


# ============================================================
# PREDICTION
# ============================================================

with st.spinner("Analyzing network traffic..."):
    attack_probability = model.predict_proba(X_imp)[:, 1]

prediction = np.where(
    attack_probability >= threshold,
    "ATTACK",
    "BENIGN",
)


# ============================================================
# RESULTS
# ============================================================

result_df = df.copy()

result_df["Attack_Probability"] = attack_probability
result_df["Prediction"] = prediction

attack_count = int(
    (prediction == "ATTACK").sum()
)

benign_count = int(
    (prediction == "BENIGN").sum()
)

total_count = len(result_df)

attack_rate = (
    attack_count / total_count * 100
    if total_count
    else 0
)


st.subheader("📊 Detection Summary")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Flows Analyzed", f"{total_count:,}")

with m2:
    st.metric(
        "Benign Flows",
        f"{benign_count:,}",
    )

with m3:
    st.metric(
        "Detected Attacks",
        f"{attack_count:,}",
    )

with m4:
    st.metric(
        "Attack Rate",
        f"{attack_rate:.2f}%",
    )


# ============================================================
# CHART
# ============================================================

chart_df = pd.DataFrame(
    {
        "Class": ["BENIGN", "ATTACK"],
        "Flows": [benign_count, attack_count],
    }
).set_index("Class")

st.bar_chart(chart_df)


# ============================================================
# HIGH-RISK FLOWS
# ============================================================

st.subheader("🚨 Highest-Risk Flows")

high_risk = (
    result_df[
        [
            "Attack_Probability",
            "Prediction",
        ]
    ]
    .sort_values(
        "Attack_Probability",
        ascending=False,
    )
    .head(20)
)

st.dataframe(
    high_risk,
    use_container_width=True,
)


# ============================================================
# FULL RESULTS
# ============================================================

with st.expander("View analyzed traffic results"):
    st.dataframe(
        result_df,
        use_container_width=True,
        height=450,
    )


# ============================================================
# DOWNLOAD
# ============================================================

csv_bytes = result_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Detection Results",
    data=csv_bytes,
    file_name="cyber_threat_detection_results.csv",
    mime="text/csv",
)


# ============================================================
# PROJECT NOTE
# ============================================================

st.divider()

st.caption(
    "Security layer: SHA-256 fingerprint → AES-256-GCM "
    "decryption → Ed25519 signature verification → inference."
)
