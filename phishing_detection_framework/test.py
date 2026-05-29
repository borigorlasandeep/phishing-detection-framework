import sys
import os
import traceback

from agents.feature_agent import FeatureExtractionAgent
from agents.ml_agent import MachineLearningAgent
from agents.decision_agent import DecisionAgent

model_dir = os.path.join(os.path.dirname(__file__), 'model', 'saved_models')
url_model_path = os.path.join(model_dir, 'url_model.pkl')
sms_model_path = os.path.join(model_dir, 'sms_model.pkl')
tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

try:
    feature_agent = FeatureExtractionAgent(tfidf_path=tfidf_path)
    ml_agent = MachineLearningAgent(url_model_path=url_model_path, sms_model_path=sms_model_path)
    decision_agent = DecisionAgent(feature_agent=feature_agent, ml_agent=ml_agent)

    decision_agent.analyze_input("hi!!", input_type="SMS")
except Exception as e:
    with open('pytest_trace.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
