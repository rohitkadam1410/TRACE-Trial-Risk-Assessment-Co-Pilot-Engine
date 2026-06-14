"""
ClinicalTrials.gov Data Ingestion and Preprocessing Pipeline
Tailored for AMD ROCm environment (Notebook-first execution).
"""
import os
import time
import logging
import requests
import pandas as pd
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

import logger_config
logger_config.setup_logging(__file__)

# ── CELL BREAK ──

def get_nested_value(d: Dict[str, Any], key: str) -> Any:
    """Recursively search for a key in a nested dictionary."""
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    for _, v in d.items():
        if isinstance(v, dict):
            res = get_nested_value(v, key)
            if res is not None:
                return res
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    res = get_nested_value(item, key)
                    if res is not None:
                        return res
    return None

def fetch_trials(conditions: List[str], target_count: int = 5000) -> List[Dict[str, Any]]:
    """Downloads clinical trial records from ClinicalTrials.gov v2 REST API."""
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    fields = "nctId,overallStatus,eligibilityCriteria,primaryOutcomes,briefSummary,officialTitle,enrollmentCount,studyType,phases,hasExpandedAccess,conditions"
    
    session = requests.Session()
    all_records = []
    
    for condition in conditions:
        if len(all_records) >= target_count:
            break
            
        page_token = None
        while len(all_records) < target_count:
            params = {
                "query.cond": condition,
                "filter.overallStatus": "COMPLETED,TERMINATED,WITHDRAWN,SUSPENDED",
                "pageSize": 100
            }
            if page_token:
                params["pageToken"] = page_token
                
            success = False
            for attempt in range(3):
                try:
                    resp = session.get(base_url, params=params, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    studies = data.get("studies", [])
                    all_records.extend(studies)
                    
                    if len(all_records) % 500 < 100:
                        logging.info(f"Collected {len(all_records)} records so far...")
                        
                    page_token = data.get("nextPageToken")
                    success = True
                    break
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
                    
            if not success:
                logging.error(f"Failed to fetch data for condition: {condition}")
                break
                
            if not page_token:
                break
                
    if len(all_records) < 2000:
        raise ValueError(
            f"Only collected {len(all_records)} records (needed >= 2000). "
            "Please check your internet connection or API status."
        )
        
    return all_records

if __name__ == "__main__":
    # Standalone test placeholder
    pass

# ── CELL BREAK ──

def parse_and_preprocess(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Parses JSON responses into a flat pandas DataFrame."""
    parsed = []
    for r in records:
        nct_id = get_nested_value(r, "nctId") or ""
        status = get_nested_value(r, "overallStatus") or ""
        
        # Text fields
        eligibility = get_nested_value(r, "eligibilityCriteria")
        eligibility_text = eligibility.get("text", "") if isinstance(eligibility, dict) else str(eligibility or "")
        
        outcomes = get_nested_value(r, "primaryOutcomes") or []
        outcome_texts = []
        if isinstance(outcomes, list):
            for o in outcomes:
                desc = o.get("description", "")
                measure = o.get("measure", "")
                outcome_texts.append(f"{measure}: {desc}")
        outcomes_str = " | ".join(outcome_texts)
        
        summary_dict = get_nested_value(r, "briefSummary")
        summary = summary_dict.get("text", "") if isinstance(summary_dict, dict) else str(summary_dict or "")
        
        title = get_nested_value(r, "officialTitle") or ""
        
        full_text = "\n".join(filter(None, [title, summary, eligibility_text, outcomes_str]))
        
        # Numeric & categorical
        enrollment = get_nested_value(r, "enrollmentCount")
        if isinstance(enrollment, dict):
            enrollment = enrollment.get("count", 0)
        enrollment = int(enrollment) if enrollment else 0
        
        phases = get_nested_value(r, "phases") or []
        phase_encoded = 0
        if isinstance(phases, list) and phases:
            p = phases[0].upper()
            if "PHASE1" in p or "PHASE 1" in p: phase_encoded = 1
            elif "PHASE2" in p or "PHASE 2" in p: phase_encoded = 2
            elif "PHASE3" in p or "PHASE 3" in p: phase_encoded = 3
            elif "PHASE4" in p or "PHASE 4" in p: phase_encoded = 4
            
        has_expanded_access = bool(get_nested_value(r, "hasExpandedAccess"))
        
        conditions_list = get_nested_value(r, "conditions") or []
        condition_count = len(conditions_list) if isinstance(conditions_list, list) else 1
        
        status_upper = status.upper()
        if status_upper in ["WITHDRAWN", "SUSPENDED"]:
            terminated = 1
        else:
            terminated = 0
            
        parsed.append({
            "nct_id": nct_id,
            "terminated": terminated,
            "full_text": full_text,
            "enrollment_count": enrollment,
            "phase_encoded": phase_encoded,
            "has_expanded_access": has_expanded_access,
            "condition_count": condition_count
        })
        
    df = pd.DataFrame(parsed)
    df = df.fillna("")
    return df

if __name__ == "__main__":
    # Standalone test placeholder
    pass

# ── CELL BREAK ──

def split_and_save(df: pd.DataFrame, output_dir: str = "data"):
    """Creates train/test split, demo subset, and saves to parquet."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_idx, test_idx = train_test_split(
        df.index, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['terminated']
    )
    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'
    
    raw_path = os.path.join(output_dir, "trials_raw.parquet")
    df.to_parquet(raw_path, index=False)
    logging.info(f"Saved {len(df)} records to {raw_path}")
    
    high_risk = df[(df['terminated'] == 1) & (df['phase_encoded'].isin([2, 3])) & (df['enrollment_count'] < 100)]
    med_risk = df[(df['terminated'] == 1) & (df['phase_encoded'] == 3) & (df['enrollment_count'].between(100, 500))]
    low_risk = df[(df['terminated'] == 0) & (df['phase_encoded'] == 3) & (df['enrollment_count'] > 500)]
    
    demo_dfs = []
    if len(high_risk) >= 5: demo_dfs.append(high_risk.sample(5, random_state=42))
    if len(med_risk) >= 5: demo_dfs.append(med_risk.sample(5, random_state=42))
    if len(low_risk) >= 5: demo_dfs.append(low_risk.sample(5, random_state=42))
    
    current_demo = pd.concat(demo_dfs) if demo_dfs else pd.DataFrame()
    needed_wildcards = 20 - len(current_demo)
    
    if needed_wildcards > 0:
        remaining = df[~df['nct_id'].isin(current_demo['nct_id']) if not current_demo.empty else df['nct_id'] == df['nct_id']]
        wildcards = remaining.sample(min(needed_wildcards, len(remaining)), random_state=42)
        demo_subset = pd.concat([current_demo, wildcards])
    else:
        demo_subset = current_demo.head(20)
        
    demo_path = os.path.join(output_dir, "demo_trials.parquet")
    demo_subset.to_parquet(demo_path, index=False)
    logging.info(f"Saved {len(demo_subset)} demo records to {demo_path}")
    
    total = len(df)
    term_pct = (df['terminated'] == 1).mean() * 100
    comp_pct = (df['terminated'] == 0).mean() * 100
    
    print("\n--- Pipeline Summary ---")
    print(f"Total records: {total}")
    print(f"Terminated: {term_pct:.1f}%")
    print(f"Completed: {comp_pct:.1f}%")
    
    if term_pct < 20 or comp_pct < 20:
        print("WARNING: Class imbalance detected (< 20% in minority class). Consider SMOTE or class weights during training.")

if __name__ == "__main__":
    # Standalone test placeholder
    pass

# ── CELL BREAK ──

def run_pipeline():
    """Main execution block to orchestrate data ingestion and prep."""
    conditions = ["cancer", "carcinoma", "tumor", "neoplasm", "lymphoma", "leukemia"]
    logging.info("Starting data ingestion...")
    records = fetch_trials(conditions, target_count=5000)
    
    logging.info("Parsing and preprocessing...")
    df = parse_and_preprocess(records)
    
    logging.info("Splitting and saving...")
    split_and_save(df)
    logging.info("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
