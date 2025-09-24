from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import re
from lime.lime_text import LimeTextExplainer
from flask import Flask, send_from_directory, jsonify

import numpy as np

# Load pickled model and vectorizer
with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

app = Flask(__name__, static_folder="../dist", static_url_path="/")

CORS(app)

# Initialize LIME explainer
explainer = LimeTextExplainer(
    class_names=['Fake', 'Real']
)

def extract_words(text):
    """Extract words from text, removing punctuation and converting to lowercase"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words

def get_lime_explanation(text, num_features=10):
    """Generate LIME explanation for the prediction"""
    try:
        # Create explanation
        exp = explainer.explain_instance(
            text, 
            pipeline.predict_proba, 
            num_features=num_features,
            top_labels=2
        )
        
        # Get explanation for the predicted class
        predicted_class = pipeline.predict([text])[0]
        explanation_list = exp.as_list(label=predicted_class)
        
        # Format explanations from [word, weight] pairs
        explanations = []
        for word_weight_pair in explanation_list:
            word = word_weight_pair[0]
            weight = word_weight_pair[1]
            
            explanations.append({
                'word': word,
                'weight': round(float(weight), 4),
                'importance': 'positive' if weight > 0 else 'negative',
                'abs_weight': round(abs(float(weight)), 4)
            })
        
        # Sort by absolute weight (most important first)
        explanations.sort(key=lambda x: x['abs_weight'], reverse=True)
        
        # Also return raw explanation for debugging
        raw_explanation = explanation_list
        
        return explanations, raw_explanation
    
    except Exception as e:
        print(f"Error generating LIME explanation: {e}")
        return [], []
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and (app.static_folder / path).exists():
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.get("news_text") or request.json.get("news_text")
    if not data:
        return jsonify({"error": "No text provided"}), 400

    # Get prediction and probabilities
    prediction = pipeline.predict([data])[0]
    probability = pipeline.predict_proba([data])[0]
    
    # Extract words from the input text
    words = extract_words(data)
    
    # Get LIME explanation
    explanation, raw_explanation = get_lime_explanation(data)
    
    # Generate explanation summary
    explanation_summary = generate_explanation_summary(explanation, prediction)

    result = {
        "prediction": "Real" if prediction == 1 else "Fake",
        "probability": {
            "Fake": round(float(probability[0]), 4),
            "Real": round(float(probability[1]), 4)
        },
        "confidence": round(float(max(probability)), 4),
        "words": words,
        "word_count": len(words),
        "explanation": explanation,
        "raw_explanation": raw_explanation,  # Original LIME format for debugging
        "explanation_summary": explanation_summary
    }

    if request.is_json:
        return jsonify(result)
    return render_template("index.html", result=result, input_text=data)

def generate_explanation_summary(explanations, prediction):
    """Generate a human-readable explanation summary"""
    if not explanations:
        return "Unable to generate explanation."
    
    prediction_text = "Real" if prediction == 1 else "Fake"
    
    # Get top positive and negative contributing words
    positive_words = [exp for exp in explanations if exp['importance'] == 'positive'][:3]
    negative_words = [exp for exp in explanations if exp['importance'] == 'negative'][:3]
    
    summary = f"This text is classified as **{prediction_text}** news. "
    
    if positive_words:
        pos_words = [f"'{word['word']}'" for word in positive_words]
        summary += f"Words that support this classification: {', '.join(pos_words)}. "
    
    if negative_words:
        neg_words = [f"'{word['word']}'" for word in negative_words]
        summary += f"Words that oppose this classification: {', '.join(neg_words)}. "
    
    # Add confidence insight
    top_word = explanations[0]
    summary += f"The most influential word is '{top_word['word']}' with a {top_word['importance']} impact."
    
    return summary

if __name__ == "__main__":
    app.run(debug=True)