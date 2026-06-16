import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ── CELL BREAK ──

def ensure_dirs():
    """Ensure output directories exist for saving artifacts and data."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

def get_feature_labels() -> dict[str, str]:
    """
    Return human-readable labels for each engineered feature.
    Used by the SHAP explainer to show readable labels to judges.
    """
    return {
        "log_enrollment": "Patient enrollment size",
        "phase_encoded": "Clinical trial phase",
        "has_expanded_access": "Has expanded access program",
        "condition_count": "Number of conditions treated",
        "title_length": "Official title length (words)",
        "criteria_length": "Eligibility criteria complexity",
        "outcome_count": "Number of primary endpoints",
        "has_age_restriction": "Has age restriction",
        "is_interventional": "Is interventional study",
        "has_placebo": "Uses placebo control",
        "has_randomized": "Is randomized",
        "has_multicenter": "Is multi-center study",
        "text_complexity": "Text complexity score",
        "enrollment_ratio": "Enrollment ratio vs phase expected",
        "criteria_unique_ratio": "Criteria vocabulary uniqueness"
    }

# ── CELL BREAK ──

def load_data(filepath: str = 'data/trials_raw.parquet') -> pd.DataFrame:
    """Load the raw dataset from pipeline.py."""
    return pd.read_parquet(filepath)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer structured features from the raw dataframe.
    Produces a feature matrix for XGBoost training.
    """
    df_feat = pd.DataFrame()
    
    # Set the target label
    if 'terminated' in df.columns:
        df_feat['terminated'] = df['terminated'].astype(int)
    else:
        # Fallback if raw data named it differently, but expected 'terminated'
        df_feat['terminated'] = 0 
        
    # 1. log_enrollment: log1p(enrollment_count) - handles skew
    # Impute missing with median per phase
    df_feat['phase_encoded'] = df.get('phase_encoded', pd.Series([2.0]*len(df))).fillna(2.0).astype(float)
    enrollment_series = df.get('enrollment_count', pd.Series([0.0]*len(df))).fillna(0.0).astype(float)
    
    # Calculate phase medians for non-zero enrollments
    valid_mask = enrollment_series > 0
    if valid_mask.any():
        phase_medians = enrollment_series[valid_mask].groupby(df_feat['phase_encoded'][valid_mask]).median()
    else:
        phase_medians = {1.0: 30, 2.0: 150, 3.0: 500, 4.0: 1000}
        
    df_feat['enrollment_count'] = [
        phase_medians.get(p, 150) if e == 0 else e 
        for e, p in zip(enrollment_series, df_feat['phase_encoded'])
    ]
    df_feat['log_enrollment'] = np.log1p(df_feat['enrollment_count'])
    
    # 2. phase_encoded: keep as-is (already populated above)
        
    # 3. has_expanded_access: cast to int
    if 'has_expanded_access' in df.columns:
        df_feat['has_expanded_access'] = df['has_expanded_access'].fillna(0).astype(int)
        
    # 4. condition_count: keep as-is
    if 'condition_count' in df.columns:
        df_feat['condition_count'] = df['condition_count'].fillna(0).astype(float)
        
    # 5. title_length: word count of officialTitle
    official_title = df.get('officialTitle', df.get('official_title', pd.Series([""]*len(df))))
    df_feat['title_length'] = official_title.fillna("").astype(str).apply(lambda x: len(x.split())).astype(float)
    
    # 6. criteria_length: word count of eligibilityCriteria
    criteria = df.get('eligibilityCriteria', df.get('eligibility_criteria', pd.Series([""]*len(df))))
    df_feat['criteria_length'] = criteria.fillna("").astype(str).apply(lambda x: len(x.split())).astype(float)
    
    # 7. outcome_count: count of semicolons in primaryOutcomeMeasure + 1
    outcome = df.get('primaryOutcomeMeasure', df.get('primary_outcome_measure', pd.Series([""]*len(df))))
    df_feat['outcome_count'] = outcome.fillna("").astype(str).apply(lambda x: x.count(';') + 1).astype(float)
    
    # 8. has_age_restriction: 1 if "years" appears in eligibilityCriteria
    df_feat['has_age_restriction'] = criteria.fillna("").astype(str).str.contains('years', case=False).astype(int)
    
    # 9. is_interventional: 1 if studyType == "INTERVENTIONAL"
    study_type = df.get('studyType', df.get('study_type', pd.Series([""]*len(df))))
    df_feat['is_interventional'] = (study_type.fillna("").astype(str).str.upper() == "INTERVENTIONAL").astype(int)
    
    # Consolidate text for text searches and TF-IDF
    full_text = df.get('full_text', pd.Series([""]*len(df))).fillna("").astype(str).str.lower()
    
    # 10. has_placebo: 1 if "placebo" in full_text
    df_feat['has_placebo'] = full_text.str.contains('placebo', na=False).astype(int)
    
    # 11. has_randomized: 1 if "randomized" or "randomised" in full_text
    df_feat['has_randomized'] = full_text.str.contains('randomized|randomised', na=False).astype(int)
    
    # 12. has_multicenter: 1 if "multicenter" or "multi-center" or "multi-site" in full_text
    df_feat['has_multicenter'] = full_text.str.contains('multicenter|multi-center|multi-site', na=False).astype(int)
    
    # 13. text_complexity: (criteria_length + title_length) / (outcome_count + 1)
    df_feat['text_complexity'] = (df_feat['criteria_length'] + df_feat['title_length']) / (df_feat['outcome_count'] + 1)
    
    # 14. enrollment_ratio: Phase expected vs actual
    phase_enrollment_expected = {1.0: 30, 2.0: 150, 3.0: 500, 4.0: 1000}
    df_feat['enrollment_ratio'] = df_feat.apply(
        lambda r: r['enrollment_count'] / phase_enrollment_expected.get(r['phase_encoded'], 150),
        axis=1
    ).astype(float)
    
    # 15. criteria_unique_ratio: unique words / total words in eligibility criteria
    def calc_density(text):
        if not text:
            return 0.0
        words = str(text).split()
        if not words:
            return 0.0
        return len(set(words)) / (len(words) + 1)
        
    df_feat['criteria_unique_ratio'] = criteria.apply(calc_density).astype(float)
    
    return df_feat, full_text

# ── CELL BREAK ──

def process_features():
    """
    Main execution pipeline for feature engineering.
    Loads data, engineers features, splits, scales, validates, and saves outputs.
    """
    ensure_dirs()
    print("Loading raw data...")
    df_raw = load_data('data/trials_raw.parquet')
    
    print("Engineering features...")
    df_feat, full_text = engineer_features(df_raw)
    
    # Stratified train/test split to maintain minimum 10% representation of terminated trials
    stratify_col = df_feat['terminated'] if 'terminated' in df_feat.columns else None
    
    train_idx, test_idx = train_test_split(
        df_feat.index, 
        test_size=0.2, 
        random_state=42, 
        stratify=stratify_col
    )
    
    df_train = df_feat.loc[train_idx].copy()
    df_test = df_feat.loc[test_idx].copy()
    text_train = full_text.loc[train_idx]
    text_test = full_text.loc[test_idx]
    
    print(f"Train size: {len(df_train)} ({len(df_train)/len(df_feat):.1%})")
    print(f"Test size: {len(df_test)} ({len(df_test)/len(df_feat):.1%})")
    
    # Scale specific continuous features on train split only
    cols_to_scale = ['log_enrollment', 'criteria_length', 'title_length', 'text_complexity', 'enrollment_ratio', 'criteria_unique_ratio']
    
    print("Fitting and applying standard scaler...")
    scaler = StandardScaler()
    df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
    df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])
    joblib.dump(scaler, 'artifacts/feature_scaler.pkl')
    
    # Data Validation Assertions
    assert not df_train.isna().any().any(), "NaN values found in train features!"
    assert not df_test.isna().any().any(), "NaN values found in test features!"
    
    train_ratio = len(df_train) / len(df_feat)
    assert 0.75 <= train_ratio <= 0.85, f"Train split ratio {train_ratio:.2f} is outside 75-85% range!"
    
    train_term_pct = df_train['terminated'].mean()
    test_term_pct = df_test['terminated'].mean()
    assert train_term_pct >= 0.05, f"Train terminated % is {train_term_pct:.1%}, needs >= 5%"
    assert test_term_pct >= 0.05, f"Test terminated % is {test_term_pct:.1%}, needs >= 5%"
    
    assert all(pd.api.types.is_numeric_dtype(df_train[col]) for col in df_train.columns), "Non-numeric features found!"
    
    # Output Correlation Matrix
    print("\nTop 5 Feature Correlations with 'terminated':")
    corr_matrix = df_train.corr()
    if 'terminated' in corr_matrix.columns:
        corrs = corr_matrix['terminated'].abs().sort_values(ascending=False).drop('terminated')
        print(corrs.head(5))
    
    # Save processed feature matrices
    print("\nSaving feature matrices...")
    df_train.to_parquet('data/features_train.parquet', index=False)
    df_test.to_parquet('data/features_test.parquet', index=False)
    
    # Extract and save feature metadata for SHAP
    meta = {
        "columns": list(df_train.columns),
        "dtypes": {col: str(dtype) for col, dtype in df_train.dtypes.items()},
        "ranges": {
            col: {"min": float(df_train[col].min()), "max": float(df_train[col].max())} 
            for col in df_train.columns
        }
    }
    with open('artifacts/feature_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
        
    # TF-IDF fallback processing
    print("Processing TF-IDF features (fallback)...")
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=5)
    tfidf_train = vectorizer.fit_transform(text_train)
    tfidf_test = vectorizer.transform(text_test)
    
    joblib.dump(vectorizer, 'artifacts/tfidf_vectorizer.pkl')
    sparse.save_npz('data/tfidf_train.npz', tfidf_train)
    sparse.save_npz('data/tfidf_test.npz', tfidf_test)
    
    print("Feature engineering complete!")

# ── CELL BREAK ──

if __name__ == "__main__":
    process_features()
