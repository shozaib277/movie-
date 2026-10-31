"""
Baltistan Tourism Recommendation Engine
Loads saved machine learning models & pipeline artifacts from models/
and provides high-performance inference for Hybrid, Content-based,
Rating-based (IMDb WR), and User-Personalized recommendations.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def minmax(s):
    """Normalize array or series to [0, 1] range safely."""
    if isinstance(s, pd.Series):
        vals = s.values
    else:
        vals = np.array(s)
    v_min = vals.min()
    v_max = vals.max()
    denom = v_max - v_min
    if denom == 0:
        return np.ones_like(vals) * 0.5
    return (vals - v_min) / (denom + 1e-9)

class BaltistanRecommender:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.load_models()

    def load_models(self):
        """Load all joblib serialized models and metadata."""
        if not os.path.exists(os.path.join(self.models_dir, 'tfidf_vectorizer.joblib')):
            from train_and_save_models import train_and_save
            train_and_save()

        self.tfidf_vectorizer = joblib.load(os.path.join(self.models_dir, 'tfidf_vectorizer.joblib'))
        self.tfidf_matrix = joblib.load(os.path.join(self.models_dir, 'tfidf_matrix.joblib'))
        self.place_similarity_matrix = joblib.load(os.path.join(self.models_dir, 'place_similarity_matrix.joblib'))
        self.places_df = joblib.load(os.path.join(self.models_dir, 'places_processed.joblib'))
        self.rating_artifacts = joblib.load(os.path.join(self.models_dir, 'rating_model.joblib'))
        self.user_data = joblib.load(os.path.join(self.models_dir, 'user_profiles.joblib'))

        with open(os.path.join(self.models_dir, 'model_metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # Index lookup maps
        self.pid_to_index = {pid: idx for idx, pid in enumerate(self.places_df['place_id'])}
        self.index_to_pid = {idx: pid for idx, pid in enumerate(self.places_df['place_id'])}

    def format_place_record(self, row, extra_metrics=None):
        """Format a place row into a clean dictionary for JSON API and templates."""
        record = {
            'place_id': str(row['place_id']),
            'name': str(row['name']),
            'district': str(row['district']),
            'category': str(row['category']),
            'interest_tags': [t.strip() for t in str(row['interest_tags']).split(';') if t.strip()],
            'activities': [a.strip() for a in str(row['activities']).split(';') if a.strip()],
            'best_season': str(row['best_season']),
            'difficulty': str(row['difficulty']),
            'altitude_m': int(row['altitude_m']) if pd.notnull(row['altitude_m']) else None,
            'latitude': float(row['latitude']) if pd.notnull(row['latitude']) else None,
            'longitude': float(row['longitude']) if pd.notnull(row['longitude']) else None,
            'access': str(row['access']),
            'description': str(row['description']),
            'avg_rating': round(float(row['avg_rating_synthetic']), 2) if 'avg_rating_synthetic' in row and pd.notnull(row['avg_rating_synthetic']) else (round(float(row['R']), 2) if 'R' in row else 4.0),
            'rating_count': int(row['rating_count_synthetic']) if 'rating_count_synthetic' in row and pd.notnull(row['rating_count_synthetic']) else (int(row['v']) if 'v' in row else 100),
            'weighted_rating': round(float(row['WR']), 2) if 'WR' in row and pd.notnull(row['WR']) else 4.0,
            'image_url': str(row['image_url']) if 'image_url' in row else "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=800&q=80"
        }
        if extra_metrics:
            record.update(extra_metrics)
        return record

    def recommend_hybrid(self, interests='', district=None, category=None, difficulty=None,
                         season=None, alpha=0.6, top_n=6, exclude_ids=None):
        """
        Hybrid recommendation:
        Final Score = alpha * Content_Score + (1 - alpha) * Rating_Score
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        df = self.places_df.copy()

        # 1. Content Score
        if interests and interests.strip():
            query_vec = self.tfidf_vectorizer.transform([interests.strip()])
            raw_content_sim = cosine_similarity(query_vec, self.tfidf_matrix).ravel()
            content_norm = minmax(raw_content_sim)
        else:
            # If no text query provided, content score defaults to uniform 0.5
            content_norm = np.ones(len(df)) * 0.5

        df['content_norm'] = content_norm

        # 2. Rating Score (IMDb WR normalized)
        df['rating_norm'] = df['WR_norm'].values

        # 3. Hybrid Final Score
        df['final_score'] = alpha * df['content_norm'] + (1.0 - alpha) * df['rating_norm']

        # Apply Filters
        if exclude_ids:
            df = df[~df['place_id'].isin(exclude_ids)]
        if district and district.lower() != 'all':
            df = df[df['district'].str.lower() == district.lower()]
        if category and category.lower() != 'all':
            df = df[df['category'].str.lower().str.contains(category.lower())]
        if difficulty and difficulty.lower() != 'all':
            df = df[df['difficulty'].str.lower() == difficulty.lower()]
        if season and season.lower() != 'all':
            df = df[df['best_season'].str.lower().str.contains(season.lower())]

        # Sort by final score descending
        df_sorted = df.sort_values('final_score', ascending=False).head(top_n)

        results = []
        for _, row in df_sorted.iterrows():
            metrics = {
                'match_percentage': int(round(row['final_score'] * 100)),
                'content_score_pct': int(round(row['content_norm'] * 100)),
                'rating_score_pct': int(round(row['rating_norm'] * 100)),
                'alpha_used': round(alpha, 2)
            }
            results.append(self.format_place_record(row, metrics))

        return results

    def get_similar_places(self, place_id, top_n=4):
        """Find places with highest item-item cosine similarity to the selected place."""
        if place_id not in self.pid_to_index:
            return []
        
        idx = self.pid_to_index[place_id]
        sim_scores = self.place_similarity_matrix[idx].copy()
        
        # Exclude self
        sim_scores[idx] = -1.0
        
        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        results = []
        for sim_idx in top_indices:
            row = self.places_df.iloc[sim_idx]
            sim_val = sim_scores[sim_idx]
            metrics = {
                'similarity_score': round(float(sim_val), 3),
                'match_percentage': int(round(float(sim_val) * 100))
            }
            results.append(self.format_place_record(row, metrics))
        return results

    def recommend_for_user(self, user_id, alpha=0.6, top_n=6):
        """
        Personalized recommendations for a tourist profile based on their
        interaction history and liked destinations.
        """
        user_meta = self.user_data['user_meta'].get(user_id, None)
        user_ratings = self.user_data['user_ratings'].get(user_id, [])

        if not user_meta and not user_ratings:
            # Fallback to hybrid with general popular tags
            return {
                'user_info': {'user_id': user_id, 'interests': 'nature;adventure', 'trip_type': 'general', 'home_city': 'Visitor'},
                'past_ratings': [],
                'recommendations': self.recommend_hybrid(interests='nature adventure photography', alpha=alpha, top_n=top_n)
            }

        rated_pids = {r['place_id'] for r in user_ratings}
        liked_pids = [r['place_id'] for r in user_ratings if r['rating'] >= 4]
        if not liked_pids:
            liked_pids = list(rated_pids)

        # Build user profile embedding from liked places TF-IDF vectors
        if liked_pids:
            valid_indices = [self.pid_to_index[p] for p in liked_pids if p in self.pid_to_index]
            if valid_indices:
                user_vector = np.asarray(self.tfidf_matrix[valid_indices].mean(axis=0))
            else:
                user_vector = np.asarray(self.tfidf_matrix.mean(axis=0))
        else:
            # Fallback to user interests text
            interests_text = user_meta.get('interests', '').replace(';', ' ') if user_meta else 'nature'
            user_vector = self.tfidf_vectorizer.transform([interests_text])

        sim_scores = cosine_similarity(user_vector, self.tfidf_matrix).ravel()
        content_norm = minmax(sim_scores)

        df = self.places_df.copy()
        df['content_norm'] = content_norm
        df['rating_norm'] = df['WR_norm'].values
        df['final_score'] = alpha * df['content_norm'] + (1.0 - alpha) * df['rating_norm']

        # Exclude places already visited/rated by user
        unseen_df = df[~df['place_id'].isin(rated_pids)]
        if len(unseen_df) < top_n:
            unseen_df = df

        top_df = unseen_df.sort_values('final_score', ascending=False).head(top_n)

        recs = []
        for _, row in top_df.iterrows():
            metrics = {
                'match_percentage': int(round(row['final_score'] * 100)),
                'content_score_pct': int(round(row['content_norm'] * 100)),
                'rating_score_pct': int(round(row['rating_norm'] * 100)),
                'alpha_used': round(alpha, 2)
            }
            recs.append(self.format_place_record(row, metrics))

        # Format past rated places with names
        past_rated_info = []
        for r in sorted(user_ratings, key=lambda x: x['rating'], reverse=True)[:6]:
            if r['place_id'] in self.pid_to_index:
                p_row = self.places_df.iloc[self.pid_to_index[r['place_id']]]
                past_rated_info.append({
                    'place_id': r['place_id'],
                    'name': p_row['name'],
                    'district': p_row['district'],
                    'rating': r['rating'],
                    'trip_type': r.get('trip_type', ''),
                    'image_url': p_row.get('image_url', '')
                })

        return {
            'user_info': {
                'user_id': user_id,
                'interests': user_meta.get('interests', '') if user_meta else '',
                'trip_type': user_meta.get('trip_type', '') if user_meta else '',
                'home_city': user_meta.get('home_city', '') if user_meta else ''
            },
            'past_ratings': past_rated_info,
            'recommendations': recs
        }

    def get_place_details(self, place_id):
        """Get full details for a single place plus similar recommendations."""
        if place_id not in self.pid_to_index:
            return None
        idx = self.pid_to_index[place_id]
        row = self.places_df.iloc[idx]
        place_data = self.format_place_record(row)
        place_data['similar_places'] = self.get_similar_places(place_id, top_n=4)
        return place_data

    def get_all_places(self, search='', district=None, category=None, difficulty=None):
        """Browse all places with live search and filter."""
        df = self.places_df.copy()
        if search and search.strip():
            q = search.strip().lower()
            mask = (
                df['name'].str.lower().str.contains(q) |
                df['description'].str.lower().str.contains(q) |
                df['interest_tags'].str.lower().str.contains(q) |
                df['activities'].str.lower().str.contains(q) |
                df['district'].str.lower().str.contains(q)
            )
            df = df[mask]

        if district and district.lower() != 'all':
            df = df[df['district'].str.lower() == district.lower()]
        if category and category.lower() != 'all':
            df = df[df['category'].str.lower().str.contains(category.lower())]
        if difficulty and difficulty.lower() != 'all':
            df = df[df['difficulty'].str.lower() == difficulty.lower()]

        df = df.sort_values('WR', ascending=False)
        return [self.format_place_record(row) for _, row in df.iterrows()]

    def get_sample_users(self, count=12):
        """Return a curated list of user personas for testing the personalized recommender."""
        users = []
        user_meta = self.user_data['user_meta']
        for uid in list(user_meta.keys())[:count]:
            meta = user_meta[uid]
            ratings_count = len(self.user_data['user_ratings'].get(uid, []))
            users.append({
                'user_id': uid,
                'interests': meta.get('interests', '').replace(';', ', '),
                'trip_type': meta.get('trip_type', '').title(),
                'home_city': meta.get('home_city', ''),
                'ratings_count': ratings_count
            })
        return users

    def get_metrics(self):
        """Return model metadata, parameters, and evaluation comparison."""
        return self.metadata
