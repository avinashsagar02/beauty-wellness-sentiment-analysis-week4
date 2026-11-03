"""
Beauty & Wellness Customer Sentiment & Feedback Analysis
Week 4: Advanced Exploratory Data Analysis & Feature Engineering
"""

import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

def engineer_beauty_features(df):
    """
    Transforms preprocessed beauty review data by creating domain-specific
    predictive features:
    1. Lexical Pattern Indicators (Adverse Reactions vs Packaging Defects)
    2. Numerical Discrepancy & Logarithmic Pricing Expectations
    3. Biological Demographics x Category Interaction Variables
    4. Statistical Mann-Whitney U Hypothesis Testing
    """
    print("=" * 70)
    print("Executing Beauty & Wellness Feature Engineering Pipeline (Week 4)")
    print("=" * 70)
    
    df = df.copy()

    # 1. Lexical Pattern Flags (Formulation Allergy vs Packaging Failure)
    allergy_regex = r'\b(acne|breakout|breakouts|burn|burning|allergic|rash|cystic|redness|peeling)\b'
    package_regex = r'\b(pump|broke|broken|leak|leaking|dispenser|dropper|lid|cap)\b'
    
    df['adverse_reaction_flag'] = df['review_text'].str.contains(allergy_regex, case=False, regex=True).astype(int)
    df['packaging_defect_flag'] = df['review_text'].str.contains(package_regex, case=False, regex=True).astype(int)

    # 2. Text Statistics & Logarithmic Pricing Ratios
    df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))
    scaler = MinMaxScaler()
    df['norm_rating'] = scaler.fit_transform(df[['rating']])
    
    price_col = df['price_usd'].fillna(df['price_usd'].median()) if 'price_usd' in df.columns else pd.Series(30.0, index=df.index)
    df['log_price'] = np.log1p(price_col)
    df['price_expectation_ratio'] = df['log_price'] / np.log1p(df['word_count'])

    # 3. Categorical Interaction: Biological Skin Type x Category
    skin_clean = df['skin_type'].fillna('Not Specified') if 'skin_type' in df.columns else pd.Series('Not Specified', index=df.index)
    cat_clean = df['category'].fillna('General') if 'category' in df.columns else pd.Series('General', index=df.index)
    df['skin_category_interaction'] = skin_clean + '_' + cat_clean

    # 4. Statistical Validation: Mann-Whitney U Test (Word Count across Polarity)
    extreme_lens = df[df['rating'].isin([1, 5])]['word_count']
    moderate_lens = df[~df['rating'].isin([1, 5])]['word_count']
    
    if len(extreme_lens) > 1 and len(moderate_lens) > 1:
        u_stat, p_val = stats.mannwhitneyu(extreme_lens, moderate_lens, alternative='two-sided')
        print(f"[STAT TEST] Mann-Whitney U Test: U-stat={u_stat:.2f}, p-value={p_val:.4f}")
    else:
        print("[STAT TEST] Mann-Whitney U Test: Sufficient group samples required.")

    print(f"[INFO] Feature Engineering Completed. Total Features: {df.shape[1]}")
    return df

def run_pipeline():
    # Representative validation batch for automated execution testing
    sample_data = [
        {"product_id": "P428230", "rating": 1.0, "skin_type": "Sensitive", "category": "Serum", "price_usd": 68.0,
         "review_text": "Gave me terrible cystic acne along my jawline within 4 days. Not good!"},
        {"product_id": "P428230", "rating": 2.0, "skin_type": "Oily", "category": "Serum", "price_usd": 68.0,
         "review_text": "Burning sensation on cheeks. Pump broke after 2 uses. Leaking bottle."},
        {"product_id": "P981244", "rating": 5.0, "skin_type": "Dry", "category": "Moisturizer", "price_usd": 54.0,
         "review_text": "Holy grail moisturizer! Absorbs instantly and restores damaged skin barrier."},
        {"product_id": "P981244", "rating": 4.0, "skin_type": "Sensitive", "category": "Moisturizer", "price_usd": 54.0,
         "review_text": "Good daily hydration but mild redness around my nose. Still okay value."},
        {"product_id": "P112349", "rating": 3.0, "skin_type": "Combination", "category": "Cleanser", "price_usd": 22.0,
         "review_text": "Average cleanser. Decent foaming action without overdrying."}
    ]
    df = pd.DataFrame(sample_data)
    processed_df = engineer_beauty_features(df)
    
    print("\n[OUTPUT SAMPLES]")
    print(processed_df[['product_id', 'skin_category_interaction', 'adverse_reaction_flag', 'packaging_defect_flag', 'price_expectation_ratio']])

if __name__ == "__main__":
    run_pipeline()
