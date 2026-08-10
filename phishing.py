import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from scipy.sparse import hstack, csr_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET = "phishing_emails.csv"

df = pd.read_csv(DATASET)

# Remove missing values
df = df.dropna(subset=["text", "label"])

# Normalize labels
df["label"] = df["label"].astype(str).str.strip().str.lower()

label_mapping = {
    "phishing": 1,
    "spam": 1,
    "malicious": 1,
    "safe": 0,
    "legitimate": 0,
    "ham": 0
}

df["target"] = df["label"].map(label_mapping)

# Remove unknown labels
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)

print("\nDataset Information")
print("-------------------")
print("Total emails:", len(df))
print("Phishing:", sum(df["target"] == 1))
print("Safe:", sum(df["target"] == 0))


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text).lower()

    # Replace URLs with a token
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " EMAIL ", text)

    # Keep alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["text"].apply(clean_text)


# ============================================================
# 3. EXTRACT URL / SECURITY FEATURES
# ============================================================

def extract_features(text):

    text = str(text)

    urls = re.findall(
        r"(https?://[^\s]+|www\.[^\s]+)",
        text.lower()
    )

    # Number of URLs
    url_count = len(urls)

    # Number of suspicious keywords
    suspicious_words = [
        "verify",
        "verification",
        "urgent",
        "password",
        "account",
        "login",
        "click",
        "suspended",
        "confirm",
        "security",
        "bank",
        "payment",
        "winner",
        "prize",
        "limited",
        "expire",
        "reset",
        "credential"
    ]

    lower_text = text.lower()

    keyword_count = sum(
        lower_text.count(word)
        for word in suspicious_words
    )

    # Number of exclamation marks
    exclamation_count = text.count("!")

    # Number of capital letters
    capital_count = sum(
        1 for char in text if char.isupper()
    )

    # Number of digits
    digit_count = sum(
        1 for char in text if char.isdigit()
    )

    # Email length
    text_length = len(text)

    # URL characteristics
    long_url_count = 0
    ip_url_count = 0
    https_count = 0
    suspicious_domain_count = 0

    suspicious_domains = [
        ".tk",
        ".ml",
        ".ga",
        ".cf",
        ".gq",
        "bit.ly",
        "tinyurl",
        "goo.gl"
    ]

    for url in urls:

        if len(url) > 75:
            long_url_count += 1

        if url.startswith("https://"):
            https_count += 1

        try:
            parsed = urlparse(
                url if url.startswith("http")
                else "http://" + url
            )

            hostname = parsed.hostname

            if hostname:
                # Detect IP address URLs
                if re.match(
                    r"^\d{1,3}(\.\d{1,3}){3}$",
                    hostname
                ):
                    ip_url_count += 1

                if any(
                    domain in hostname
                    for domain in suspicious_domains
                ):
                    suspicious_domain_count += 1

        except Exception:
            pass

    return [
        url_count,
        keyword_count,
        exclamation_count,
        capital_count,
        digit_count,
        text_length,
        long_url_count,
        ip_url_count,
        https_count,
        suspicious_domain_count
    ]


feature_names = [
    "url_count",
    "keyword_count",
    "exclamation_count",
    "capital_count",
    "digit_count",
    "text_length",
    "long_url_count",
    "ip_url_count",
    "https_count",
    "suspicious_domain_count"
]

numeric_features = np.array(
    [extract_features(text) for text in df["text"]]
)

print("\nExtracted Features:")
print(feature_names)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = \
    train_test_split(
        df["clean_text"],
        numeric_features,
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"]
    )


# ============================================================
# 5. TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    stop_words="english"
)

X_train_text = vectorizer.fit_transform(X_text_train)
X_test_text = vectorizer.transform(X_text_test)


# ============================================================
# 6. SCALE NUMERIC FEATURES
# ============================================================

scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_num_train)
X_test_num = scaler.transform(X_num_test)

X_train_num = csr_matrix(X_train_num)
X_test_num = csr_matrix(X_test_num)


# ============================================================
# 7. COMBINE TEXT + URL FEATURES
# ============================================================

X_train = hstack([
    X_train_text,
    X_train_num
])

X_test = hstack([
    X_test_text,
    X_test_num
])


# ============================================================
# 8. TRAIN MACHINE LEARNING MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ============================================================
# 9. PREDICTION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("PHISHING EMAIL DETECTOR")
print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Safe", "Phishing"]
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Phishing"],
    yticklabels=["Safe", "Phishing"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Phishing Email Detection - Confusion Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 11. PREDICT NEW EMAIL
# ============================================================

def predict_email(email):

    cleaned = clean_text(email)

    # TF-IDF
    text_vector = vectorizer.transform([cleaned])

    # Numeric features
    numeric = np.array(
        [extract_features(email)]
    )

    numeric = scaler.transform(numeric)

    numeric = csr_matrix(numeric)

    # Combine
    final_features = hstack([
        text_vector,
        numeric
    ])

    prediction = model.predict(
        final_features
    )[0]

    probability = model.predict_proba(
        final_features
    )[0]

    if prediction == 1:
        result = "PHISHING"
        confidence = probability[1]
    else:
        result = "SAFE"
        confidence = probability[0]

    return result, confidence


# ============================================================
# 12. INTERACTIVE EMAIL TESTING
# ============================================================

print("\n==============================")
print("TEST A NEW EMAIL")
print("==============================")

while True:

    email = input(
        "\nEnter email text (or type 'exit'): "
    )

    if email.lower() == "exit":
        print("Program terminated.")
        break

    result, confidence = predict_email(email)

    print("\nPrediction:", result)
    print(
        f"Confidence: {confidence * 100:.2f}%"
    )