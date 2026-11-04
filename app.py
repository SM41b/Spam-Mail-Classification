
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Mail Classifier",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Mail Classification System")
st.write("An ML-based application to identify spam and legitimate emails.")


# --------------------------------------------------
# LOAD DATA, MODEL AND VECTORIZER
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("mail_data.csv")


@st.cache_resource
def load_model():
    return joblib.load("spam_mail_model.pkl")


@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")


raw_data = load_data()
model = load_model()
vectorizer = load_vectorizer()


# Prepare dataset for evaluation and insights
data = raw_data.copy()
data = data.dropna(subset=["Category", "Message"])

data["Category"] = data["Category"].str.lower().str.strip()

data["Category"] = data["Category"].map({
    "spam": 0,
    "ham": 1
})

data = data.dropna(subset=["Category"]).copy()
data["Category"] = data["Category"].astype(int)

# Create message length column
data["Message_Length"] = data["Message"].astype(str).apply(len)

# Separate features and target
X = data["Message"].astype(str)
Y = data["Category"]

# Use the same split settings as the training script
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=3,
    stratify=Y
)

# Transform test messages using the saved vectorizer
X_test_features = vectorizer.transform(X_test)

# Generate predictions
Y_test_prediction = model.predict(X_test_features)

# Calculate evaluation metrics
accuracy = accuracy_score(Y_test, Y_test_prediction)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Spam Mail Prediction",
        "Model Performance",
        "Dataset Insights"
    ]
)


# ==================================================
# SECTION 1: SPAM MAIL PREDICTION
# ==================================================

if page == "Spam Mail Prediction":

    st.header("📩 Spam Mail Prediction")

    st.write(
        "Enter an email message below to check whether "
        "it is Spam or Ham."
    )

    input_mail = st.text_area(
        "Enter your email message:",
        placeholder="Paste your email message here...",
        height=180
    )

    if st.button("Classify Email", type="primary"):

        if input_mail.strip() == "":
            st.warning("Please enter an email message.")

        else:
            # Convert message into TF-IDF features
            input_features = vectorizer.transform([input_mail])

            # Predict email category
            prediction = model.predict(input_features)[0]

            # Get prediction probability
            probabilities = model.predict_proba(input_features)[0]

            # Find probability corresponding to predicted class
            predicted_probability = probabilities[
                list(model.classes_).index(prediction)
            ]

            if prediction == 0:
                st.error("🚨 This email is classified as SPAM.")
            else:
                st.success("✅ This email is classified as HAM.")

            st.metric(
                "Model Confidence",
                f"{predicted_probability * 100:.2f}%"
            )

            st.caption(
                "Confidence represents the model's predicted probability, "
                "not a guarantee that the classification is correct."
            )


# ==================================================
# SECTION 2: MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.header("📊 Model Performance")

    st.write(
        "Evaluation of the Logistic Regression model "
        "on the testing dataset."
    )

    # Calculate classification metrics
    report = classification_report(
        Y_test,
        Y_test_prediction,
        labels=[0, 1],
        target_names=["Spam", "Ham"],
        output_dict=True,
        zero_division=0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Spam Precision", f"{report['Spam']['precision'] * 100:.2f}%")
    col3.metric("Spam Recall", f"{report['Spam']['recall'] * 100:.2f}%")
    col4.metric("Spam F1-Score", f"{report['Spam']['f1-score'] * 100:.2f}%")

    st.subheader("Classification Report")

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        Y_test,
        Y_test_prediction,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Spam", "Ham"])
    ax.set_yticklabels(["Spam", "Ham"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Testing Dataset")

    st.write(f"Total testing samples: **{len(Y_test)}**")
    st.write(f"Correct predictions: **{(Y_test == Y_test_prediction).sum()}**")
    st.write(f"Incorrect predictions: **{(Y_test != Y_test_prediction).sum()}**")


# ==================================================
# SECTION 3: DATASET INSIGHTS
# ==================================================

elif page == "Dataset Insights":

    st.header("📈 Dataset Insights")

    st.write(
        "Explore the distribution and characteristics "
        "of the email messages used to train the model."
    )

    # Dataset KPIs
    total_messages = len(data)
    spam_count = (data["Category"] == 0).sum()
    ham_count = (data["Category"] == 1).sum()

    spam_percentage = (spam_count / total_messages) * 100
    ham_percentage = (ham_count / total_messages) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Messages", f"{total_messages:,}")
    col2.metric("Spam Messages", f"{spam_count:,}")
    col3.metric("Ham Messages", f"{ham_count:,}")

    # Message distribution
    st.subheader("Spam vs Ham Distribution")

    distribution = pd.DataFrame({
        "Category": ["Spam", "Ham"],
        "Number of Messages": [spam_count, ham_count]
    })

    st.bar_chart(
        distribution.set_index("Category")
    )

    col1, col2 = st.columns(2)

    col1.metric("Spam Percentage", f"{spam_percentage:.2f}%")
    col2.metric("Ham Percentage", f"{ham_percentage:.2f}%")

    # Message length analysis
    st.subheader("Message Length Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Average message length by category")

        avg_length = (
            data.groupby("Category")["Message_Length"]
            .mean()
            .rename(index={0: "Spam", 1: "Ham"})
        )

        st.bar_chart(avg_length)

    with col2:
        st.write("Message length statistics")

        length_stats = (
            data.groupby("Category")["Message_Length"]
            .agg(["mean", "median", "min", "max"])
            .rename(index={0: "Spam", 1: "Ham"})
        )

        st.dataframe(length_stats.round(2), use_container_width=True)

    # Dataset preview
    st.subheader("Dataset Preview")

    display_data = data.copy()
    display_data["Category"] = display_data["Category"].map({
        0: "Spam",
        1: "Ham"
    })

    st.dataframe(
        display_data[["Category", "Message", "Message_Length"]].head(10),
        use_container_width=True
    )

    # Descriptive statistics
    st.subheader("Dataset Summary")

    st.dataframe(
        data[["Message_Length"]].describe().round(2),
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Spam Mail Classification | Logistic Regression + TF-IDF"
)
