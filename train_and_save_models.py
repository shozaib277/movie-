"""
Baltistan Tourism Recommendation System - Model Training & Artifact Generator
Trains Content-Based (TF-IDF + Cosine Similarity), Rating-Based (IMDb Weighted Rating),
and Hybrid Recommender models, evaluates them on train/test split, and serializes
all model artifacts with joblib for production Flask serving.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Curated high quality imagery for all 27 Baltistan tourist places
PLACE_IMAGES = {
    "P001": "https://images.unsplash.com/photo-1589182373726-e4f658ab50f0?auto=format&fit=crop&w=800&q=80", # Shangrila
    "P002": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", # Upper Kachura
    "P003": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&w=800&q=80", # Satpara Lake
    "P004": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=800&q=80", # Deosai
    "P005": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80", # Sheosar Lake
    "P006": "https://images.unsplash.com/photo-1599818828588-ef22a90098df?auto=format&fit=crop&w=800&q=80", # Kharpocho Fort
    "P007": "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=800&q=80", # Sarfaranga Desert
    "P008": "https://images.unsplash.com/photo-1432405972618-c60b0225b8f9?auto=format&fit=crop&w=800&q=80", # Manthokha Waterfall
    "P009": "https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=800&q=80", # Basho Valley
    "P010": "https://images.unsplash.com/photo-1544644181-1484b3fdfc62?auto=format&fit=crop&w=800&q=80", # Shigar Fort
    "P011": "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?auto=format&fit=crop&w=800&q=80", # Shigar Valley
    "P012": "https://images.unsplash.com/photo-1582650625119-3a31f8418b7d?auto=format&fit=crop&w=800&q=80", # Khaplu Palace
    "P013": "https://images.unsplash.com/photo-1564507592333-c60657eea523?auto=format&fit=crop&w=800&q=80", # Chaqchan Mosque
    "P014": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?auto=format&fit=crop&w=800&q=80", # Khaplu Valley
    "P015": "https://images.unsplash.com/photo-1486870591958-9b9d0d1dda99?auto=format&fit=crop&w=800&q=80", # Hushe Valley
    "P016": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=800&q=80", # Masherbrum Base Camp
    "P017": "https://images.unsplash.com/photo-1454496522488-7a8e488e8606?auto=format&fit=crop&w=800&q=80", # K2 Base Camp / Baltoro
    "P018": "https://images.unsplash.com/photo-1519681393784-d120267933ba?auto=format&fit=crop&w=800&q=80", # Askole Village
    "P019": "https://images.unsplash.com/photo-1483921020237-2ff51e8e4b22?auto=format&fit=crop&w=800&q=80", # Biafo Glacier Trek
    "P020": "https://images.unsplash.com/photo-1517411032315-54ef2cb783bb?auto=format&fit=crop&w=800&q=80", # Chogo Lungma Glacier
    "P021": "https://images.unsplash.com/photo-1544161515-4ab6ce6db874?auto=format&fit=crop&w=800&q=80", # Chutron Hot Springs
    "P022": "https://images.unsplash.com/photo-1548013146-72479768bada?auto=format&fit=crop&w=800&q=80", # Buddha Rock
    "P023": "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=800&q=80", # Skardu Bazaar
    "P024": "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?auto=format&fit=crop&w=800&q=80", # Rondu Valley
    "P025": "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?auto=format&fit=crop&w=800&q=80", # Kharmang Valley
    "P026": "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=800&q=80", # Shyok River
    "P027": "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=800&q=80"  # Machulu Village
}

def minmax(s):
    """Normalize a series or array to 0..1 scale"""
    denom = s.max() - s.min()
    if denom == 0:
        return np.zeros_like(s)
    return (s - s.min()) / (denom + 1e-9)

def build_feature_text(row):
    """Build enriched text representation for TF-IDF content filtering"""
    return ' '.join([
        str(row['category']),
        str(row['district']),
        str(row['difficulty']),
        str(row['best_season']),
        (str(row['interest_tags']).replace(';', ' ') + ' ') * 3,
        (str(row['activities']).replace(';', ' ') + ' ') * 2,
        str(row['description'])
    ])

def train_and_save():
    print("=" * 60)
    print("Baltistan Tourism Recommender - Training & Saving Models")
    print("=" * 60)

    # 1. Load Datasets
    places = pd.read_csv(os.path.join(BASE_DIR, 'baltistan_places.csv'))
    ratings = pd.read_csv(os.path.join(BASE_DIR, 'baltistan_ratings_SYNTHETIC.csv'))
    users = pd.read_csv(os.path.join(BASE_DIR, 'baltistan_users_SYNTHETIC.csv'))

    print(f"Loaded: {len(places)} places, {len(ratings)} ratings, {len(users)} users.")

    # Attach images
    places['image_url'] = places['place_id'].map(PLACE_IMAGES).fillna(
        "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=800&q=80"
    )

    # 2. Rating Model (IMDb Weighted Rating)
    # WR = (v / (v + m)) * R + (m / (v + m)) * C
    stats = ratings.groupby('place_id').rating.agg(R='mean', v='count').reset_index()
    C_global_mean = float(ratings.rating.mean())
    m_threshold = float(stats.v.quantile(0.60))

    stats['WR'] = (stats.v / (stats.v + m_threshold)) * stats.R + (m_threshold / (stats.v + m_threshold)) * C_global_mean
    stats['WR_norm'] = minmax(stats['WR'])

    # Merge stats back to places dataframe
    places_processed = places.merge(stats[['place_id', 'R', 'v', 'WR', 'WR_norm']], on='place_id', how='left')
    places_processed['R'] = places_processed['R'].fillna(C_global_mean)
    places_processed['v'] = places_processed['v'].fillna(0)
    places_processed['WR'] = places_processed['WR'].fillna(C_global_mean)
    places_processed['WR_norm'] = places_processed['WR_norm'].fillna(0.5)

    print(f"Rating Model: Global Mean C = {C_global_mean:.3f}, Minimum Votes m (60th %ile) = {m_threshold:.1f}")

    # 3. Content-Based Model (TF-IDF + Cosine Similarity)
    places_processed['feature_text'] = places_processed.apply(build_feature_text, axis=1)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(places_processed['feature_text'])
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape} (Vocabulary: {len(tfidf_vectorizer.vocabulary_)} features)")

    # Precalculate place-to-place cosine similarity matrix
    place_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 4. Train/Test Evaluation (Precision@5, Recall@5, Alpha tuning)
    print("Evaluating models using 80/20 train/test split...")
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    train_stats = train_ratings.groupby('place_id').rating.agg(R='mean', v='count').reset_index()
    Cg = float(train_ratings.rating.mean())
    mm = float(train_stats.v.quantile(0.60))
    train_stats['WR'] = (train_stats.v / (train_stats.v + mm)) * train_stats.R + (mm / (train_stats.v + mm)) * Cg

    WR_train_series = train_stats.set_index('place_id').WR.reindex(places_processed.place_id).fillna(Cg)
    WR_norm_train = pd.Series(minmax(WR_train_series).values, index=places_processed.place_id)

    pid_index = {p: i for i, p in enumerate(places_processed.place_id)}
    K = 5

    def rec_lists_eval(uid, alpha=0.6):
        seen = set(train_ratings[train_ratings.user_id == uid].place_id)
        liked = train_ratings[(train_ratings.user_id == uid) & (train_ratings.rating >= 4)].place_id
        if len(liked) == 0:
            liked = train_ratings[train_ratings.user_id == uid].place_id
        if len(liked) == 0:
            prof = np.asarray(tfidf_matrix.mean(axis=0))
        else:
            prof = np.asarray(tfidf_matrix[[pid_index[p] for p in liked if p in pid_index]].mean(axis=0))
        
        sim_scores = cosine_similarity(prof, tfidf_matrix).ravel()
        c_series = pd.Series(minmax(pd.Series(sim_scores)).values, index=places_processed.place_id)
        
        cand = [p for p in places_processed.place_id if p not in seen]
        if not cand:
            cand = list(places_processed.place_id)
            
        rating_only = WR_norm_train[cand].sort_values(ascending=False).index[:K].tolist()
        content_only = c_series[cand].sort_values(ascending=False).index[:K].tolist()
        hyb = (alpha * c_series + (1 - alpha) * WR_norm_train)[cand].sort_values(ascending=False).index[:K].tolist()
        return rating_only, content_only, hyb

    eval_results = {'Rating only': [], 'Content only': [], 'Hybrid': []}
    for uid in test_ratings.user_id.unique():
        relevant = set(test_ratings[(test_ratings.user_id == uid) & (test_ratings.rating >= 4)].place_id)
        if not relevant:
            continue
        r_rec, c_rec, h_rec = rec_lists_eval(uid, alpha=0.6)
        for name, recs in [('Rating only', r_rec), ('Content only', c_rec), ('Hybrid', h_rec)]:
            hit = len(relevant & set(recs))
            eval_results[name].append((hit / K, hit / len(relevant)))

    summary_metrics = {}
    for k, v in eval_results.items():
        summary_metrics[k] = {
            'Precision@5': float(round(np.mean([a for a, b in v]), 4)),
            'Recall@5': float(round(np.mean([b for a, b in v]), 4))
        }

    # Alpha curve tuning
    alpha_curve = []
    for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        precisions = []
        for uid in test_ratings.user_id.unique():
            relevant = set(test_ratings[(test_ratings.user_id == uid) & (test_ratings.rating >= 4)].place_id)
            if not relevant:
                continue
            recs = rec_lists_eval(uid, alpha=a)[2]
            precisions.append(len(relevant & set(recs)) / K)
        alpha_curve.append({
            'alpha': float(a),
            'precision': float(round(np.mean(precisions), 4))
        })

    print("Model Evaluation Summary (Precision@5 & Recall@5):")
    for model_name, metrics in summary_metrics.items():
        print(f"  - {model_name:12}: Precision@5 = {metrics['Precision@5']}, Recall@5 = {metrics['Recall@5']}")

    # 5. Save Artifacts with joblib
    print("\nSaving model artifacts to disk...")

    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    joblib.dump(tfidf_matrix, os.path.join(MODELS_DIR, 'tfidf_matrix.joblib'))
    joblib.dump(place_similarity_matrix, os.path.join(MODELS_DIR, 'place_similarity_matrix.joblib'))
    joblib.dump(places_processed, os.path.join(MODELS_DIR, 'places_processed.joblib'))

    rating_model_artifacts = {
        'C_global_mean': C_global_mean,
        'm_threshold': m_threshold,
        'place_stats': stats.set_index('place_id').to_dict(orient='index')
    }
    joblib.dump(rating_model_artifacts, os.path.join(MODELS_DIR, 'rating_model.joblib'))

    # Save user profiles and history
    user_ratings_history = ratings.groupby('user_id').apply(
        lambda df: df[['place_id', 'rating', 'visit_month', 'trip_type']].to_dict(orient='records')
    ).to_dict()

    user_meta = users.set_index('user_id').to_dict(orient='index')
    joblib.dump({'user_meta': user_meta, 'user_ratings': user_ratings_history}, os.path.join(MODELS_DIR, 'user_profiles.joblib'))

    # Save JSON metadata & evaluation metrics
    metadata = {
        'model_name': 'Baltistan Recommender AI System',
        'version': '1.0.0',
        'total_places': len(places_processed),
        'total_ratings': len(ratings),
        'total_users': len(users),
        'rating_parameters': {
            'global_mean_C': round(C_global_mean, 3),
            'min_votes_m': round(m_threshold, 1)
        },
        'evaluation_metrics': summary_metrics,
        'alpha_tuning_curve': alpha_curve,
        'districts': sorted(places_processed['district'].unique().tolist()),
        'categories': sorted(places_processed['category'].unique().tolist()),
        'difficulties': ['Easy', 'Moderate', 'Hard'],
        'top_interest_tags': [
            'adventure', 'nature', 'photography', 'culture', 'camping', 'family',
            'trekking', 'heritage', 'lakes', 'boating', 'glacier'
        ]
    }

    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("SUCCESS: All models and artifacts saved in 'models/' directory!")
    print(f"Files saved in {MODELS_DIR}:")
    for f in os.listdir(MODELS_DIR):
        fsize = os.path.getsize(os.path.join(MODELS_DIR, f))
        print(f"  - {f} ({fsize} bytes)")

if __name__ == '__main__':
    train_and_save()
