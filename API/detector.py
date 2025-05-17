import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from lime.lime_text import LimeTextExplainer
import numpy as np

import re
import nltk
import tldextract
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

##############################################################################################################
# Phishing Email Detector
class PhishingEmailDetector:
    def __init__(self, classifier: BaseEstimator = LogisticRegression(solver='liblinear', class_weight='balanced')):
        """
        Initialize the detector with a classifier.
        :param classifier: Classifier to use (default: LogisticRegression)
        """
        # nltk.download('stopwords')
        # stop_words = set(stopwords.words('english'))

        self.text_col = 'combined_text'
        self.meta_cols = ['same_domain', 'num_links', 'num_suspicious_words']
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        self.scaler = StandardScaler()
        # self.classifier = classifier or LogisticRegression(solver='liblinear', class_weight='balanced')
        self.classifier = classifier
        self.pipeline = None

        # if self.classifier is None:
        #     self.classifier = LogisticRegression(solver='liblinear', class_weight='balanced')

    #---------------------------------------------------------------------------------------------------------
    # Text Cleaning
    def _clean_text(self, text):
        """
        Clean the input text by removing HTML tags, URLs, and non-alphanumeric characters.
        Convert to lowercase and remove stop words.
        :param text: The text to clean
        :return: Cleaned text
        """
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"http\S+", "LINKURL", text)
        # text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text) # convert to lowercase
        text = re.sub(r"\d+", "NUM", text) # remove numbers
        text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
        return " ".join([w for w in text.split() if w not in stop_words])

    #---------------------------------------------------------------------------------------------------------
    # Metadata
    def _extract_domain(self, email):
        if pd.isnull(email) or "@" not in email:
            return "unknown"
        return email.split('@')[-1].lower()

    def _is_same_domain(self, sender, recipient):
        return int(self._extract_domain(sender) == self._extract_domain(recipient))

    def _count_links(self, text):
        if pd.isnull(text):
            return 0
        return len(re.findall(r"http[s]?://", text))

    def _count_suspicious_words(self, text, keywords=None):
        if pd.isnull(text):
            return 0
        if keywords is None:
            keywords = ["verify", "login", "click", "update", "urgent", "password"]
        return sum(word in text.lower() for word in keywords)

    #---------------------------------------------------------------------------------------------------------
    # Preprocessing
    def preprocess_dataframe(self, dfx, debug=False):
        """
        Preprocess the input DataFrame by cleaning text and extracting metadata.
        :param dfx: Input DataFrame with columns 'subject', 'body', 'sender', 'recipient'
        :param debug: If True, print debug information
        :return: Processed DataFrame with cleaned text and metadata
        """
        df = dfx.copy()
        df['clean_subject'] = df['subject'].apply(self._clean_text)
        df['clean_body'] = df['body'].apply(self._clean_text)
        df[self.text_col] = df['clean_subject'] + " " + df['clean_body']
        df['same_domain'] = df.apply(lambda x: self._is_same_domain(x['sender'], x['recipient']), axis=1)
        df['num_links'] = df['body'].apply(self._count_links)
        df['num_suspicious_words'] = df['body'].apply(self._count_suspicious_words)

        if debug:
            print("Cleaned Text:")
            print(df[self.text_col].head())
            print("Metadata:")
            print(df[self.meta_cols].head())

        return df[[self.text_col] + self.meta_cols]

    #---------------------------------------------------------------------------------------------------------
    # Build Pipeline
    def _build_pipeline(self):
        transformer = ColumnTransformer([
            ('tfidf', self.vectorizer, self.text_col),
            ('meta', self.scaler, self.meta_cols)
        ])
        return Pipeline([
            ('features', transformer),
            ('clf', self.classifier)
        ])

    #---------------------------------------------------------------------------------------------------------
    # Train Model
    def fit(self, df, labels, debug=False):
        """
        Fit the model to the training data.
        :param df: Input DataFrame with columns 'subject', 'body', 'sender', 'recipient'
        :param labels: Labels for the training data (1 for phishing, 0 for legitimate)
        :param debug: If True, print debug information
        :return: None
        """
        processed = self.preprocess_dataframe(df, debug=debug)

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(processed, labels)

    #---------------------------------------------------------------------------------------------------------
    # Predict Labels
    def predict(self, df, debug=False):
        """
        Predict labels for the input DataFrame.
        :param df: Input DataFrame with columns 'subject', 'body', 'sender', 'recipient'
        :param debug: If True, print debug information
        :return: Predicted labels (1 for phishing, 0 for legitimate)
        """
        if self.pipeline is None:
            raise ValueError("You must fit the model before calling predict.")

        processed = self.preprocess_dataframe(df, debug=debug)
        return self.pipeline.predict(processed)

    def predict_proba(self, df):
        if self.pipeline is None:
            raise ValueError("You must fit the model before calling predict_proba.")

        processed = self.preprocess_dataframe(df)
        return self.pipeline.predict_proba(processed)
    
    def predict_single(self, subject, body, sender, recipient):
        """
        Predicts if a single email is phishing or not.
        :param subject: Email subject
        :param body: Email body
        :param sender: Sender's email address
        :param recipient: Recipient's email address
        :return: Tuple (label, probability)
        """
        df = pd.DataFrame({
            'subject': [subject],
            'body': [body],
            'sender': [sender],
            'recipient': [recipient]
        })
        
        lables = self.predict(df)
        preds = self.predict_proba(df)
        return lables[0], preds[0][1]  # Return label and probability of phishing

    #---------------------------------------------------------------------------------------------------------
    # Evaluate
    def evaluate(self, df, labels):
        """
        Evaluate the model on the test data. 
        Prints classification report.
        :param df: Input DataFrame with columns 'subject', 'body', 'sender', 'recipient'
        :param labels: True labels for the test data (1 for phishing, 0 for legitimate)
        :return: None
        """
        preds = self.predict(df)
        print(classification_report(labels, preds))

    #---------------------------------------------------------------------------------------------------------
    # Get pipeline for advanced use
    def get_pipeline(self):
        return self.pipeline

    def explain_instance(self, df_row, num_features=10):
        """
        Explains a single email instance (df_row must be a one-row DataFrame).
        Returns explanation object with weights.
        """
        if self.pipeline is None:
            raise ValueError("You must fit the model before calling explain_instance.")

        # Prepare LimeTextExplainer
        class_names = ['Legitimate', 'Phishing']
        explainer = LimeTextExplainer(class_names=class_names)

        # Extract text from row
        email_text = self._clean_text(df_row['subject'].values[0]) + " " + self._clean_text(df_row['body'].values[0])

        # Build a prediction function that LIME can use
        def predict_proba(texts):
            # texts: list of raw texts from LIME
            temp_df = pd.DataFrame({
                'subject': [''] * len(texts),
                'body': texts,
                'sender': [df_row['sender'].values[0]] * len(texts),
                'recipient': [df_row['recipient'].values[0]] * len(texts)
            })
            return self.predict_proba(temp_df)

        # Explain
        exp = explainer.explain_instance(email_text, predict_proba, num_features=num_features)
        return exp
    
    def explain_instance_as_list(self, df_row, num_features=10):
        """
        Explains a single email instance (df_row must be a one-row DataFrame).
        Returns explanation object with weights.
        """
        exp = self.explain_instance(df_row, num_features=num_features)
        return exp.as_list()  # List of tuples (feature, weight)
    
    def explain_single_instance_as_list(self, subject, body, sender, recipient, num_features=10):
        """
        Explains a single email instance (df_row must be a one-row DataFrame).
        Returns explanation object with weights.
        """
        df = pd.DataFrame({
            'subject': [subject],
            'body': [body],
            'sender': [sender],
            'recipient': [recipient]
        })
        
        exp = self.explain_instance_as_list(df.iloc[[0]], num_features=num_features)
        return exp
    
    #---------------------------------------------------------------------------------------------------------
    # Save and Load Model
    def save_model(self, model_path):
        """
        Save the trained model to a file.
        """
        if self.pipeline is None:
            raise ValueError("You must fit the model before calling save_model.")
        joblib.dump(self.pipeline, model_path)
    
    def load_model(self, model_path):
        """
        Load a pre-trained model from a file.
        """
        self.pipeline = joblib.load(model_path)