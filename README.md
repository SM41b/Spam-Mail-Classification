# Spam Mail Classification

## Live Demo
[Click here to try the Spam Mail Classification App](YOUR_DEPLOYED_APP_URL)

## Project Overview
This project is a Machine Learning-based Spam Mail Classification system that identifies whether an email message is **Spam** or **Ham (Legitimate)**.

The system uses Natural Language Processing (NLP) techniques, TF-IDF (Term Frequency-Inverse Document Frequency) for text feature extraction, and Logistic Regression for classification.

A Streamlit web application provides an interactive interface where users can enter email messages, receive predictions, explore model performance, and analyze dataset insights.

## Project Objectives
- Classify email messages as Spam or Ham.
- Apply text preprocessing and feature extraction using TF-IDF.
- Train a Logistic Regression classification model.
- Evaluate model performance using classification metrics.
- Build an interactive web application using Streamlit.

## Dataset
The project uses a labelled SMS/email dataset stored in `mail_data.csv`.

**Dataset information:**
- Total records: 5,572
- Features:
  - `Category`: Email classification label (spam or ham).
  - `Message`: Email message text.
- Target classes:
  - `0`: Spam
  - `1`: Ham

## Data Preprocessing
The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Replaced missing values with empty strings.
3. Converted categorical labels into numerical values:
   - Spam → 0
   - Ham → 1
4. Separated the message column and target labels.
5. Split the dataset into training (80%) and testing (20%) sets.

## Exploratory Data Analysis
The dataset was explored to understand its structure and characteristics, including:
- Dataset dimensions and structure.
- Spam and Ham message distribution.
- Message length statistics.
- Comparison of message characteristics across categories.

## Feature Extraction
**TF-IDF Vectorization** was used to convert text messages into numerical feature vectors.

Configuration:
- `min_df = 1`
- `stop_words = 'english'`
- `lowercase = True`

TF-IDF assigns importance to words based on their frequency within a message and across the dataset.

## Model Development
**Algorithm:** Logistic Regression

The Logistic Regression classifier was trained using the TF-IDF-transformed training messages.

**Train-test split:**
- Training data: 80%
- Testing data: 20%
- Random state: 3
- Stratified sampling

The trained model and fitted TF-IDF vectorizer were saved using Joblib for reuse in the Streamlit application.

## Model Evaluation

The model was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

| Metric | Training / Testing Result |
|---|---:|
| Training Accuracy | 96.68% |
| Testing Accuracy | 97.13% |
| Spam Precision | 99% |
| Spam Recall | 79% |
| Spam F1-Score | 88% |
| Ham Precision | 97% |
| Ham Recall | 100% |

### Confusion Matrix

| Actual / Predicted | Spam | Ham |
|---|---:|---:|
| Spam | 118 | 31 |
| Ham | 1 | 965 |

**Key Observation:** The model achieves 97.13% testing accuracy and performs particularly well at identifying legitimate messages. However, its 79% spam recall indicates that some spam messages are incorrectly classified as legitimate emails.

## Streamlit Application Features

The application contains three sections:

### 1. Spam Mail Prediction
- Enter an email message.
- Predict whether the message is Spam or Ham.
- View the model's prediction confidence.

### 2. Model Performance
- View training and testing model evaluation results.
- Explore accuracy, precision, recall, and F1-score.
- View the classification report.
- Visualize the confusion matrix.
- Examine correct and incorrect predictions.

### 3. Dataset Insights
- View total email messages.
- Explore Spam vs Ham distribution.
- Compare Spam and Ham percentages.
- Analyze average message length.
- Explore message length statistics.
- Preview dataset records.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **NLP / Feature Extraction:** TF-IDF Vectorizer
- **Classification Algorithm:** Logistic Regression
- **Model Serialization:** Joblib
- **Data Visualization:** Matplotlib
- **Web Application:** Streamlit

## Project Structure

```text
Spam Mail Classification/
│
├── Spam Mail Classification.py
├── app.py
├── mail_data.csv
├── spam_mail_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd "spam mail classification"
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```powershell
.\venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Disclaimer
This project is developed for educational and demonstration purposes. Predictions are based on a trained machine learning model and should not be considered a guaranteed determination of whether an email is malicious or legitimate.