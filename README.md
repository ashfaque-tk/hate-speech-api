# Hate Speech Detection API

A production-ready FastAPI service that detects hate speech using a calibrated LinearSVC model.

# Model
Simple TF-IDF with Linear SVC

## Features
- **Calibrated Probabilities:** Uses Platt Scaling to provide confidence scores.
- **Uncertainty Logging:** Automatically logs predictions with 0.3-0.7 confidence for human review.
- **Dockerized:** Fully containerized for easy deployment.

## Model Analysis
During stress testing, the model achieved ~75% accuracy. A known limitation was identified where the TF-IDF vectorizer struggles with sarcasm. Further, it cannot distinguish between general rudeness and targeted hate (it will lean towards hate).
providing a clear case for moving to Transformer-based models in future iterations.
