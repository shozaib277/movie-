import json
import os

with open('Baltistan_Recommender_Task.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check if save models cell already exists
already_has_save = any('joblib' in ''.join(c.get('source', [])) for c in nb['cells'])

if not already_has_save:
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Save Models & Pipeline Artifacts\n",
            "Serialize the trained TF-IDF vectorizer, place feature matrix, precomputed similarity, weighted ratings, and processed metadata using `joblib` for deployment in the Flask web application."
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import joblib\n",
            "import json\n",
            "\n",
            "os.makedirs('models', exist_ok=True)\n",
            "\n",
            "# Save TF-IDF Vectorizer and Matrix\n",
            "joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')\n",
            "joblib.dump(X, 'models/tfidf_matrix.joblib')\n",
            "\n",
            "# Save places processed dataframe & similarity matrix\n",
            "joblib.dump(places, 'models/places_processed.joblib')\n",
            "joblib.dump(cosine_similarity(X, X), 'models/place_similarity_matrix.joblib')\n",
            "\n",
            "# Save rating statistics\n",
            "rating_artifacts = {\n",
            "    'C_global_mean': float(C_mean),\n",
            "    'm_threshold': float(m),\n",
            "    'place_stats': stats.set_index('place_id').to_dict(orient='index')\n",
            "}\n",
            "joblib.dump(rating_artifacts, 'models/rating_model.joblib')\n",
            "\n",
            "print('All models and artifacts saved successfully into models/ directory!')\n"
        ]
    }
    
    nb['cells'].append(markdown_cell)
    nb['cells'].append(code_cell)
    
    with open('Baltistan_Recommender_Task.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Updated Baltistan_Recommender_Task.ipynb with model saving cells.")
else:
    print("Notebook already contains model saving logic.")
