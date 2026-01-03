import pandas as pd
import requests
import time
from tqdm import tqdm
import json
import os

API_url = "http://localhost:8000/predict"
test_data_path = f"{os.getcwd()}/data/labeled_regression_hate_speech.json"



def run_stress_tests():

    with open(test_data_path) as f:
        cases = json.load(f)

    mismatch = 0

    for case in cases:
        resp = requests.post(API_url, json={"text": case["Content"]})
        result = resp.json()
        pred =  result['label']
        conf = result['confidence']

        if pred != case["Label"]:
            mismatch += 1
            print(f"MISMATCH\n"
                f"text: {case['Content']}\n"
                f"true: {case['Label']} | pred: {pred}\n"
                f"confidence:{conf}\n")

    print(f"Label mismatches: {mismatch}/{len(cases)}")


if __name__ =='__main__':
    run_stress_tests()