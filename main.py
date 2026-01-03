from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.nlp_pipeline.preprocess import PreprocessText
import json

app = FastAPI(title="Hate Speech Detection API")
model = joblib.load("model/hate_speech_model.joblib")

# logging function or low confidence predictions
def log_low_confidence(text:str,proba:list):
    """Logs requests where the model is uncertain (hate/non_hate b/w 0.3 and 0.7)
    saves to a json file for future analysis"""
    hate_prob = float(proba[0])
    print(hate_prob)
    if 0.3 <= hate_prob  <= 0.7:
        log_entry = {"text":text,"hate speech probability":round(hate_prob,4)
                     ,"none_hate_probablity":round(float(proba[0]),4)}
        with open("uncertain_requests.jsonl","a") as f:
            f.write(json.dumps(log_entry)+"\n")

class TextInput(BaseModel):
    text : str

class PredicionOutput(BaseModel):
    label: int
    label_name: str
    confidence: float
    probabilities: dict

@app.post("/predict",response_model=PredicionOutput)
def predict_hate_speech(input:TextInput):
    text_list = input.text
    pred = model.predict(text_list)
    proba = model.predict_proba(text_list)[0]

    confidence_score = float(max(proba))
    #log low confidence score
    log_low_confidence(input.text,proba)

    return {"label": int(pred),
            "label_name":"hate_speech" if pred==1 else "none_hate",
            "confidence": round(confidence_score,4),
            "probabilities":{"none_hate":round(float(proba[0]),4),
                            "hate_speech":round(float(proba[1]),4)}}
@app.get("/")
def health_check():
    return {"status":"ok"}