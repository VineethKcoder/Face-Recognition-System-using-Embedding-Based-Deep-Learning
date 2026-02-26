import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(layout="wide")
st.title("Face Recognition Evaluation Dashboard")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    embeddings = np.load("baseline_embeddings.npy")
    labels = np.load("baseline_labels.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    return embeddings, labels, train_labels, test_labels

embeddings, labels, train_labels, test_labels = load_data()

# -----------------------------------------------------
# SIDEBAR - DATASET INFO
# -----------------------------------------------------
st.sidebar.header("Dataset Overview")
st.sidebar.write("Total Samples:", embeddings.shape[0])
st.sidebar.write("Embedding Dimension:", embeddings.shape[1])
st.sidebar.write("Total Identities:", len(np.unique(labels)))

# -----------------------------------------------------
# DATA LEAKAGE CHECK
# -----------------------------------------------------
st.sidebar.subheader("Data Leakage Check")

train_ids = set(train_labels)
test_ids = set(test_labels)
leakage = train_ids.intersection(test_ids)

if len(leakage) == 0:
    st.sidebar.success("No Identity Leakage Detected")
else:
    st.sidebar.error(f"Leakage Detected: {leakage}")

# -----------------------------------------------------
# SPLIT EMBEDDINGS (STRICT TEST EVALUATION)
# -----------------------------------------------------
train_mask = np.isin(labels, train_labels)
test_mask = np.isin(labels, test_labels)

train_embeddings = embeddings[train_mask]
test_embeddings = embeddings[test_mask]
test_labels_full = labels[test_mask]

# Normalize test embeddings
test_embeddings = normalize(test_embeddings)

# -----------------------------------------------------
# ROC COMPUTATION
# -----------------------------------------------------
st.header("ROC Analysis (Evaluation Set Only)")

with st.spinner("Computing similarity matrix..."):

    similarity_matrix = cosine_similarity(test_embeddings)

    sim_scores = []
    y_true = []

    n = len(test_labels_full)

    for i in range(n):
        for j in range(i + 1, n):

            sim_scores.append(similarity_matrix[i, j])

            if test_labels_full[i] == test_labels_full[j]:
                y_true.append(1)
            else:
                y_true.append(0)

    sim_scores = np.array(sim_scores)
    y_true = np.array(y_true)

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_true, sim_scores)
roc_auc = auc(fpr, tpr)

# Optimal Threshold (Youden’s J)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

st.write(f"AUC Score: **{roc_auc:.4f}**")
st.write(f"Optimal Threshold: **{optimal_threshold:.4f}**")

# -----------------------------------------------------
# PLOT ROC CURVE
# -----------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name=f"ROC Curve (AUC = {roc_auc:.3f})"
))

fig.add_shape(
    type='line',
    x0=0, y0=0,
    x1=1, y1=1,
    line=dict(dash='dash')
)

fig.update_layout(
    title="Receiver Operating Characteristic",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# SIMILARITY DISTRIBUTION
# -----------------------------------------------------
st.header("Similarity Score Distribution")

same_scores = sim_scores[y_true == 1]
diff_scores = sim_scores[y_true == 0]

hist_fig = go.Figure()

hist_fig.add_trace(go.Histogram(
    x=same_scores,
    name="Same Identity",
    opacity=0.6
))

hist_fig.add_trace(go.Histogram(
    x=diff_scores,
    name="Different Identity",
    opacity=0.6
))

hist_fig.update_layout(
    barmode='overlay',
    title="Similarity Score Distribution",
    xaxis_title="Cosine Similarity",
    yaxis_title="Count",
    height=600
)

st.plotly_chart(hist_fig, use_container_width=True)

st.success("Evaluation Completed Using Strict Test Set Only.")