"""
Baltistan Tourism AI Recommender - Flask Web Application
Provides interactive REST API and ultra-modern web UI for exploring Baltistan
destinations, personalized hybrid AI recommendations, and model analytics.
"""

import os
from flask import Flask, render_template, request, jsonify
from recommender import BaltistanRecommender

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize the recommender engine
recommender = BaltistanRecommender()

@app.route('/')
def index():
    """Main application landing page."""
    metrics = recommender.get_metrics()
    sample_users = recommender.get_sample_users(count=8)
    initial_places = recommender.get_all_places()[:8]
    return render_template(
        'index.html',
        metrics=metrics,
        sample_users=sample_users,
        initial_places=initial_places,
        districts=metrics.get('districts', []),
        categories=metrics.get('categories', []),
        difficulties=metrics.get('difficulties', []),
        tags=metrics.get('top_interest_tags', [])
    )

@app.route('/api/places', methods=['GET'])
def get_places():
    """Get all places with optional search and filters."""
    search = request.args.get('search', '')
    district = request.args.get('district', 'all')
    category = request.args.get('category', 'all')
    difficulty = request.args.get('difficulty', 'all')
    places = recommender.get_all_places(
        search=search,
        district=district,
        category=category,
        difficulty=difficulty
    )
    return jsonify({'status': 'success', 'count': len(places), 'places': places})

@app.route('/api/place/<place_id>', methods=['GET'])
def get_place(place_id):
    """Get complete place details and similar places."""
    place = recommender.get_place_details(place_id)
    if not place:
        return jsonify({'status': 'error', 'message': f'Place {place_id} not found'}), 404
    return jsonify({'status': 'success', 'place': place})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Compute Hybrid, Content-based, or Rating-based recommendations."""
    data = request.get_json() or {}
    interests = data.get('interests', '')
    district = data.get('district', 'all')
    category = data.get('category', 'all')
    difficulty = data.get('difficulty', 'all')
    season = data.get('season', 'all')
    alpha = float(data.get('alpha', 0.6))
    top_n = int(data.get('top_n', 6))
    exclude = data.get('exclude', [])

    recs = recommender.recommend_hybrid(
        interests=interests,
        district=district,
        category=category,
        difficulty=difficulty,
        season=season,
        alpha=alpha,
        top_n=top_n,
        exclude_ids=exclude
    )
    return jsonify({
        'status': 'success',
        'count': len(recs),
        'alpha': alpha,
        'recommendations': recs
    })

@app.route('/api/recommend/user/<user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """Personalized recommendations for a specific tourist persona."""
    alpha = float(request.args.get('alpha', 0.6))
    top_n = int(request.args.get('top_n', 6))
    result = recommender.recommend_for_user(user_id=user_id, alpha=alpha, top_n=top_n)
    return jsonify({'status': 'success', 'data': result})

@app.route('/api/similar/<place_id>', methods=['GET'])
def similar_places(place_id):
    """Item-to-item similarity based on TF-IDF cosine similarity."""
    top_n = int(request.args.get('top_n', 4))
    sims = recommender.get_similar_places(place_id, top_n=top_n)
    return jsonify({'status': 'success', 'count': len(sims), 'similar_places': sims})

@app.route('/api/users', methods=['GET'])
def get_users():
    """Retrieve sample tourist user personas."""
    count = int(request.args.get('count', 12))
    users = recommender.get_sample_users(count=count)
    return jsonify({'status': 'success', 'users': users})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return model evaluation metrics and alpha tuning curves."""
    metrics = recommender.get_metrics()
    return jsonify({'status': 'success', 'metrics': metrics})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Baltistan Tourism Recommender on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
