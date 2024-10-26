import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import boto3

# Initialize CloudWatch client
cloudwatch = boto3.client('cloudwatch', region_name='your-aws-region')  # replace with our AWS region

# Loading out  Dataset
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
    
    # Split into training  and testining sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    
    # Convert text data into TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Train Model
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Evaluate Model and Send Alert if Accuracy Falls Below Threshold
def evaluate_model(model, X_test, y_test, threshold=0.9):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    print(classification_report(y_test, predictions))
    
    # Check accuracy and send alert if below threshold
    if accuracy < threshold:
        send_alert_to_cloudwatch(accuracy)
    return accuracy

# Send Alert to CloudWatch
def send_alert_to_cloudwatch(accuracy):
    cloudwatch.put_metric_data(
        Namespace='SpamClassifier',
        MetricData=[
            {
                'MetricName': 'SpamDetectionAccuracy',
                'Dimensions': [
                    {
                        'Name': 'Model',
                        'Value': 'NaiveBayes'
                    },
                ],
                'Value': accuracy,  
                'Unit': 'None'
            },
        ]
    )
    print("CloudWatch Alert: Accuracy fell below threshold")

# Run the Spam Classifier Pipeline
def main():
    # Load and preprocess data
    data = load_data(r"C:\Users\kalam\Downloads\archive (17)\spam.csv")  # Replace with your dataset path
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model and set CloudWatch alert if necessary
    accuracy = evaluate_model(model, X_test, y_test, threshold=0.9)

if __name__ == '__main__':
    main()
