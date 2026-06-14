import pandas as pd
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO)

from explainer import generate_demo_cache
from trainer import _load_feature_names, _load_feature_labels
from copilot import generate_demo_cache as generate_copilot_cache

def main():
    print("Loading data...")
    demo_trials = pd.read_parquet("data/demo_trials.parquet")
    model = joblib.load("artifacts/xgb_model.pkl")
    scaler = joblib.load("artifacts/feature_scaler.pkl")
    explainer = joblib.load("artifacts/shap_explainer.pkl")
    feature_names = _load_feature_names()
    feature_labels = _load_feature_labels()
    
    with open("artifacts/optimal_threshold.json") as f:
        threshold = json.load(f)["threshold"]
        
    print("Generating Explainer Cache...")
    generate_demo_cache(
        demo_trials=demo_trials,
        model=model,
        scaler=scaler,
        explainer=explainer,
        feature_names=feature_names,
        feature_labels=feature_labels,
        threshold=threshold,
        output_path="demo/demo_cache.json"
    )
    
    # print("Generating Copilot Cache...")
    # generate_copilot_cache(
    #     explainer_cache_path="demo/demo_cache.json",
    #     output_path="artifacts/demo_cache.json"
    # )
    print("Done!")

if __name__ == "__main__":
    main()
