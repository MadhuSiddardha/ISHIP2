import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email Configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT_SSL = 465
EMAIL_ADDRESS = 'madhusiddardhakala@gmail.com'  # Your email address
EMAIL_PASSWORD = 'ocdc omse gubt vrbe'  # Use the app password if 2FA is enabled

# Function to send spam notification email
def send_spam_notification(recipient_email, message_text):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = 'Spam Alert: Suspicious Email Detected'

        body = f"Warning! A suspicious email was detected as spam:\n\n{message_text}"
        msg.attach(MIMEText(body, 'plain'))

        # Use SMTP_SSL for secure connection
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT_SSL, timeout=20) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"Spam notification sent to {recipient_email}")
    except smtplib.SMTPAuthenticationError:
        print("Failed to send email: Authentication error. Check your email address or app password.")
    except smtplib.SMTPConnectError:
        print("Failed to send email: Connection error. Check your network and SMTP server details.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Load Dataset
def load_data(filepath):
    data = pd.read_csv(filepath, encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

# Preprocess Data
def preprocess_data(data):
    X = data['text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Train Model
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    print(classification_report(y_test, predictions))
    return accuracy

# Predict if an email is spam and notify
def predict_and_notify(model, vectorizer, email_text, recipient_email):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)

    if prediction == 1:  # Spam detected
        print("Spam detected!")
        send_spam_notification(recipient_email, email_text)
    else:
        print("Not spam. No notification sent.")

# Run the Spam Classifier Pipeline
def main():
    data = load_data(r"C:\Users\kalam\Downloads\archive (17)\spam.csv")  # Update with actual path
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data)

    model = train_model(X_train, y_train)

    accuracy = evaluate_model(model, X_test, y_test)

    # Input email text and recipient email for real-time spam detection
    email_text = input("Enter the email text to classify: ")
    recipient_email = input("Enter the recipient email address to send spam notification if spam is detected: ")

    # Predict if the email is spam and notify recipient if it is
    predict_and_notify(model, vectorizer, email_text, recipient_email)

if __name__ == '__main__':
    main()
