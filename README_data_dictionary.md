# Baltistan Tourism Dataset (Starter Version)

## Important: what is real and what is synthetic
- **baltistan_places.csv**: 27 real tourist places across Skardu, Shigar, Ghanche (Khaplu), Kharmang and Rondu. Categories, seasons, difficulty and descriptions are written from general knowledge. **Latitude, longitude and altitude are approximate; verify them on Google Maps / OpenStreetMap before final submission.**
- **baltistan_users_SYNTHETIC.csv** and **baltistan_ratings_SYNTHETIC.csv**: 200 simulated tourists and about 2,800 simulated ratings. They are NOT real tourist reviews. They exist so the content-based, rating-filter and hybrid models can be built and tested now.
- Columns `avg_rating_synthetic` and `rating_count_synthetic` in places are computed from the synthetic ratings only.

To make it a fully real dataset: collect real ratings with a Google Form survey of tourists, guides and students (same columns as the ratings file), plus permitted public review data, then replace the synthetic files and recompute the two rating columns. State this clearly in the final report.

## places
place_id, name, district, category, interest_tags (semicolon list), activities, best_season, difficulty (Easy/Moderate/Hard), altitude_m (approx), latitude, longitude (approx), access, description, avg_rating_synthetic, rating_count_synthetic

## users
user_id, interests (semicolon list: adventure, nature, culture, photography, family, camping), trip_type, home_city

## ratings
user_id, place_id, rating (1 to 5), visit_month, trip_type, data_source ("synthetic" for all rows here; use "survey" or "public_review" for real rows)

## How it maps to the proposal
- Content-based filtering: use interest_tags, category, activities, best_season, difficulty and description with TF-IDF + cosine similarity.
- Rating filtering: weighted rating WR = v/(v+m)*R + m/(v+m)*C using the ratings file.
- Hybrid: Final Score = alpha * content score + (1 - alpha) * normalized WR.
