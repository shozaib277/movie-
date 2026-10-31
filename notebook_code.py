import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

places  = pd.read_csv('baltistan_places.csv')
ratings = pd.read_csv('baltistan_ratings_SYNTHETIC.csv')
users   = pd.read_csv('baltistan_users_SYNTHETIC.csv')

print(places.shape, ratings.shape, users.shape)
places.head()

# --- CELL ---

# Quick exploration
print(ratings.rating.value_counts().sort_index())
print()
print(places.district.value_counts())

ratings.rating.value_counts().sort_index().plot(kind='bar', title='Rating distribution')
plt.xlabel('Rating'); plt.ylabel('Count'); plt.show()

# --- CELL ---

stats = ratings.groupby('place_id').rating.agg(R='mean', v='count').reset_index()
C_mean = ratings.rating.mean()
m = stats.v.quantile(0.60)

stats['WR'] = (stats.v/(stats.v+m))*stats.R + (m/(stats.v+m))*C_mean
rating_table = places.merge(stats, on='place_id').sort_values('WR', ascending=False)

print('C (global mean) =', round(C_mean,3), '| m =', round(m,1))
rating_table[['name','district','category','R','v','WR']].head(10).round(2)

# --- CELL ---

def recommend_by_rating(n=5, district=None, category=None, min_rating=3.5, exclude=None):
    df = rating_table[rating_table.R >= min_rating]
    if district: df = df[df.district.str.lower() == district.lower()]
    if category: df = df[df.category.str.contains(category, case=False)]
    if exclude:  df = df[~df.place_id.isin(exclude)]
    return df[['place_id','name','district','category','R','v','WR']].head(n).round(2)

print('Top 5 overall');         display(recommend_by_rating(5))
print('Top 3 in Ghanche');      display(recommend_by_rating(3, district='Ghanche'))
print('Top 3 lakes');           display(recommend_by_rating(3, category='Lake'))

# --- CELL ---

def make_text(r):
    # repeat interest tags and activities so they carry more weight than the long description
    return ' '.join([
        r.category, r.district, r.difficulty, r.best_season,
        (r.interest_tags.replace(';',' ') + ' ') * 3,
        (r.activities.replace(';',' ') + ' ') * 2,
        r.description])

places['text'] = places.apply(make_text, axis=1)
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = tfidf.fit_transform(places.text)
print('TF-IDF matrix:', X.shape)

# --- CELL ---

def recommend_by_content(interests, n=5, exclude=None):
    """interests: text such as 'trekking camping glacier adventure'"""
    q = tfidf.transform([interests])
    sim = cosine_similarity(q, X).ravel()
    out = places[['place_id','name','district','category']].copy()
    out['content_score'] = sim
    if exclude: out = out[~out.place_id.isin(exclude)]
    return out.sort_values('content_score', ascending=False).head(n).round(3)

print('Adventure tourist');    display(recommend_by_content('adventure trekking camping glacier'))
print('Culture + photography');display(recommend_by_content('culture heritage fort history photography'))
print('Family, relaxed');      display(recommend_by_content('family lake easy picnic boating'))

# --- CELL ---

def similar_places(place_name, n=5):
    i = places.index[places.name.str.contains(place_name, case=False)][0]
    sim = cosine_similarity(X[i], X).ravel()
    out = places[['name','district','category']].copy()
    out['similarity'] = sim
    return out.drop(index=i).sort_values('similarity', ascending=False).head(n).round(3)

print('Places similar to Shangrila');  display(similar_places('Shangrila'))

# --- CELL ---

def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def recommend_hybrid(interests, n=5, alpha=0.6, exclude=None, district=None, season=None):
    q = tfidf.transform([interests])
    c = pd.Series(cosine_similarity(q, X).ravel(), index=places.place_id)
    w = rating_table.set_index('place_id').WR.reindex(places.place_id)
    df = places[['place_id','name','district','category','best_season']].copy()
    df['content'] = minmax(c).values
    df['rating']  = minmax(w).values
    df['final_score'] = alpha*df.content + (1-alpha)*df.rating
    if exclude:  df = df[~df.place_id.isin(exclude)]
    if district: df = df[df.district.str.lower() == district.lower()]
    if season:   df = df[df.best_season.str.contains(season, case=False)]
    return df.sort_values('final_score', ascending=False).head(n).round(3)

print('Hybrid: adventure tourist');  display(recommend_hybrid('adventure trekking camping glacier'))
print('Hybrid: culture tourist, Ghanche only'); display(recommend_hybrid('culture heritage history photography', district='Ghanche', n=3))

# --- CELL ---

from sklearn.model_selection import train_test_split

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# weighted rating from TRAIN only (avoids leakage)
st = train.groupby('place_id').rating.agg(R='mean', v='count').reset_index()
Cg, mm = train.rating.mean(), st.v.quantile(0.60)
st['WR'] = (st.v/(st.v+mm))*st.R + (mm/(st.v+mm))*Cg
WR = st.set_index('place_id').WR.reindex(places.place_id).fillna(Cg)
WRn = minmax(WR)

pid_index = {p:i for i,p in enumerate(places.place_id)}
K = 5

def rec_lists(uid, alpha=0.6):
    seen = set(train[train.user_id==uid].place_id)
    liked = train[(train.user_id==uid)&(train.rating>=4)].place_id
    if len(liked)==0: liked = train[train.user_id==uid].place_id
    prof = np.asarray(X[[pid_index[p] for p in liked]].mean(axis=0))
    c = pd.Series(cosine_similarity(prof, X).ravel(), index=places.place_id)
    cn = minmax(c)
    cand = [p for p in places.place_id if p not in seen]
    rating_only  = WRn[cand].sort_values(ascending=False).index[:K]
    content_only = cn[cand].sort_values(ascending=False).index[:K]
    hyb = (alpha*cn + (1-alpha)*WRn)[cand].sort_values(ascending=False).index[:K]
    return rating_only, content_only, hyb

res = {'Rating only':[], 'Content only':[], 'Hybrid':[]}
for uid in test.user_id.unique():
    relevant = set(test[(test.user_id==uid)&(test.rating>=4)].place_id)
    if not relevant: continue
    for name, recs in zip(res.keys(), rec_lists(uid)):
        hit = len(relevant & set(recs))
        res[name].append((hit/K, hit/len(relevant)))

summary = pd.DataFrame({k:{'Precision@5':np.mean([a for a,b in v]), 'Recall@5':np.mean([b for a,b in v])} for k,v in res.items()}).T.round(3)
summary

# --- CELL ---

summary.plot(kind='bar', rot=0, title='Model comparison (higher is better)')
plt.ylabel('Score'); plt.show()

# --- CELL ---

# Tune alpha for the hybrid model
rows = []
for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    p = []
    for uid in test.user_id.unique():
        relevant = set(test[(test.user_id==uid)&(test.rating>=4)].place_id)
        if not relevant: continue
        recs = rec_lists(uid, alpha=a)[2]
        p.append(len(relevant & set(recs))/K)
    rows.append((a, np.mean(p)))
alpha_df = pd.DataFrame(rows, columns=['alpha','Precision@5']).round(3)
display(alpha_df)
alpha_df.plot(x='alpha', y='Precision@5', marker='o', title='Effect of alpha on hybrid precision'); plt.show()

# --- CELL ---

my_interests = 'photography culture heritage family'
recommend_hybrid(my_interests, n=5, alpha=0.6)