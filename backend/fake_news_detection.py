# fake_news_pipeline.py
import numpy as np
import pandas as pd
import string
import nltk
import joblib
import warnings
warnings.filterwarnings('ignore')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lime
import lime.lime_text

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer for text preprocessing"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def remove_punct(self, text):
        return "".join([char for char in text if char not in string.punctuation])

    def count_punct_words(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        punct_count = sum(1 for char in text if char in string.punctuation)
        return round(punct_count / len(words), 3) * 100

    def count_cap_words(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        cap_count = sum(1 for char in text if char.isupper())
        return round(cap_count / len(words), 3) * 100

    def preprocess_text(self, text):
        # Clean and tokenize
        clean_text = self.remove_punct(text.lower())
        tokens = word_tokenize(clean_text)

        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word)
                 for word in tokens if word not in self.stop_words]

        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            texts = X.values
        else:
            texts = X

        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Calculate features
        features = []
        for i, text in enumerate(texts):
            body_len = len(text) - text.count(' ')
            punct_per_word = self.count_punct_words(text)
            cap_per_word = self.count_cap_words(text)

            features.append({
                'text': processed_texts[i],
                'body_len': body_len,
                'punct_per_word': punct_per_word,
                'cap_per_word': cap_per_word
            })

        return features

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract TF-IDF features and combine with engineered features"""

    def __init__(self, max_features=5000):
        self.tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=max_features)
        self.scaler = MinMaxScaler()
        self.feature_names = None

    def fit(self, X, y=None):
        # X is list of feature dictionaries
        texts = [item['text'] for item in X]

        # Fit TF-IDF
        self.tfidf.fit(texts)

        # Create feature matrix for scaling
        tfidf_features = self.tfidf.transform(texts).toarray()
        numeric_features = np.array([[item['body_len'], item['punct_per_word'],
                                    item['cap_per_word']] for item in X])

        combined_features = np.hstack([tfidf_features, numeric_features])

        # Fit scaler
        self.scaler.fit(combined_features)

        # Store feature names
        tfidf_names = list(self.tfidf.get_feature_names_out())
        numeric_names = ['body_len', 'punct_per_word', 'cap_per_word']
        self.feature_names = tfidf_names + numeric_names

        return self

    def transform(self, X):
        texts = [item['text'] for item in X]

        # Transform texts
        tfidf_features = self.tfidf.transform(texts).toarray()
        numeric_features = np.array([[item['body_len'], item['punct_per_word'],
                                    item['cap_per_word']] for item in X])

        combined_features = np.hstack([tfidf_features, numeric_features])

        # Scale features
        scaled_features = self.scaler.transform(combined_features)

        return scaled_features

class FakeNewsDetector:
    """Complete fake news detection system with LIME explanations"""

    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.pipeline = None
        self.explainer = None
        self.is_fitted = False

    def build_pipeline(self):
        """Build the complete pipeline"""
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor()

        if self.model_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('features', feature_extractor),
            ('classifier', classifier)
        ])

        return self.pipeline

    def fit(self, texts, labels):
        """Train the model"""
        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(texts, labels)

        # Initialize LIME explainer
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=["Fake", "Real"]
          )

        self.is_fitted = True
        return self

    def predict(self, texts):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.pipeline.predict_proba(texts)

    def explain_instance(self, text, num_features=10):
        """Explain a single prediction using LIME"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explaining")

        # Create a wrapper function for LIME
        def predict_proba_wrapper(texts):
            return self.pipeline.predict_proba(texts)

        explanation = self.explainer.explain_instance(
            text,
            predict_proba_wrapper,
            num_features=num_features,
            labels=[0, 1]
        )

        return explanation

    def save_model(self, filepath):
        """Save the trained model components"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save only the pipeline and necessary components
        model_data = {
            'pipeline': self.pipeline,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)

    @staticmethod
    def load_model(filepath):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        
        # Recreate the detector
        detector = FakeNewsDetector(model_type=model_data['model_type'])
        detector.pipeline = model_data['pipeline']
        detector.is_fitted = model_data['is_fitted']
        
        # Reinitialize LIME explainer
        if detector.is_fitted:
            detector.explainer = lime.lime_text.LimeTextExplainer(
                class_names=["Fake", "Real"]
            )
        
        return detector

# Training function
def train_model(fake_csv_path, true_csv_path, model_type='logistic', test_size=0.2):
    """Train the fake news detection model"""

    # Load data
    fake_df = pd.read_csv(fake_csv_path)
    true_df = pd.read_csv(true_csv_path)

    # Add labels
    fake_df["label"] = 0
    true_df["label"] = 1

    # Merge datasets
    merged_news = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        merged_news['text'],
        merged_news['label'],
        test_size=test_size,
        random_state=42,
        stratify=merged_news['label']
    )

    # Initialize and train model
    detector = FakeNewsDetector(model_type=model_type)
    detector.fit(X_train, y_train)

    # Make predictions
    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)

    # Print results
    print(f"\n{model_type.upper()} MODEL RESULTS:")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Show example explanation
    test_texts = X_test.tolist()
    sample_text = test_texts[0]
    sample_prediction = detector.predict([sample_text])[0]
    sample_proba = detector.predict_proba([sample_text])[0]

    print(f"\nSAMPLE EXPLANATION:")
    print("="*40)
    print(f"Text preview: {sample_text[:200]}...")
    print(f"Prediction: {'Real' if sample_prediction == 1 else 'Fake'}")
    print(f"Confidence: {max(sample_proba):.4f}")

    # Get LIME explanation
    explanation = detector.explain_instance(sample_text, num_features=10)
    print(f"\nTop features influencing prediction:")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:.4f}")

    return detector, X_test, y_test

if __name__ == "__main__":
    detector = FakeNewsDetector.load_model('model.pkl')
    explanation = detector.explain_instance("sciencetist discover the vaccine of covid")
    print(explanation.as_list())
    print(detector.predict(["sciencetist discover the vaccine of covid"]))