import streamlit as st
import pandas as pd
import ast
import os
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# change these paths if you move the dataset somewhere else
MOVIES_CSV ="tmdb_5000_movies.csv"
CREDITS_CSV ="tmdb_5000_credits.csv"

# paste your tmdb api key here to get movie posters
# get one free at https://www.themoviedb.org/settings/api
TMDB_API_KEY ="YOUR_TMDB_API_KEY_HERE"

POSTER_BASE ="https://image.tmdb.org/t/p/w500"
NO_POSTER ="https://via.placeholder.com/500x750?text=No+Poster"


st.set_page_config(page_title="Movie Recommender", page_icon="", layout="wide")


# styling — dark netflix-like theme
st.markdown("""
<style>
 .stApp { background-color: #0e1117; }

 .big-title {
 font-size: 2.8rem;
 font-weight: 700;
 text-align: center;
 color: #e50914;
 margin-bottom: 0;
 }
 .tagline {
 text-align: center;
 color: #888;
 margin-bottom: 2rem;
 }
 .card {
 background: #1a1d27;
 border-radius: 10px;
 padding: 10px;
 text-align: center;
 border: 1px solid #2a2d3a;
 height: 100%;
 }
 .card img {
 width: 100%;
 border-radius: 7px;
 }
 .card .name {
 font-size: 0.82rem;
 font-weight: 600;
 color: #fff;
 margin: 7px 0 4px;
 }
 .badge {
 background: #e50914;
 color: white;
 font-size: 0.72rem;
 font-weight: 700;
 padding: 2px 9px;
 border-radius: 20px;
 }
 .gtag {
 display: inline-block;
 background: #2a2d3a;
 color: #999;
 font-size: 0.62rem;
 padding: 2px 7px;
 border-radius: 8px;
 margin: 2px 1px;
 }
 .stTextInput input {
 background: #1a1d27 !important;
 color: white !important;
 border: 1px solid #e50914 !important;
 border-radius: 7px !important;
 }
 .stButton > button {
 background: #e50914;
 color: white;
 border: none;
 border-radius: 7px;
 font-weight: 600;
 width: 100%;
 padding: 0.45rem 0;
 }
 .stButton > button:hover { background: #c4070f; }
</style>
""", unsafe_allow_html=True)


# grab poster from tmdb, fall back to placeholder if anything goes wrong
def get_poster(movie_id):
 if TMDB_API_KEY =="YOUR_TMDB_API_KEY_HERE":
 return NO_POSTER
 try:
 res = requests.get(
 f"https://api.themoviedb.org/3/movie/{movie_id}",
 params={"api_key": TMDB_API_KEY}, timeout=5
 ).json()
 path = res.get("poster_path")
 return POSTER_BASE + path if path else NO_POSTER
 except:
 return NO_POSTER


def get_genres(genre_str):
 try:
 return [g["name"] for g in ast.literal_eval(str(genre_str))]
 except:
 return []


def make_soup(row):
 def names(field, limit=None):
 try:
 items = [i["name"].replace("","") for i in ast.literal_eval(str(field))]
 return items[:limit] if limit else items
 except:
 return []

 def director(crew):
 try:
 return [m["name"].replace("","") for m in ast.literal_eval(str(crew)) if m.get("job") =="Director"]
 except:
 return []

 parts = (
 director(row["crew"]) * 3
 + names(row["genres"]) * 2
 + names(row["cast"], 5)
 + names(row["keywords"])
 + str(row["overview"]).lower().split()
 )
 return"".join(parts)


@st.cache_resource(show_spinner="Loading movies...")
def load_model():
 # use cached files if they exist so it loads faster next time
 if os.path.exists("similarity_matrix.pkl") and os.path.exists("movies_df.pkl"):
 with open("similarity_matrix.pkl","rb") as f:
 sim = pickle.load(f)
 with open("movies_df.pkl","rb") as f:
 df = pickle.load(f)
 return df, sim

 movies = pd.read_csv(MOVIES_CSV)
 credits = pd.read_csv(CREDITS_CSV)

 if"movie_id" in credits.columns:
 credits.rename(columns={"movie_id":"id"}, inplace=True)

 df = movies.merge(credits, on="id")
 df = df[["id","title_x","overview","genres","keywords","cast","crew"]]
 df.rename(columns={"title_x":"title"}, inplace=True)
 df.dropna(inplace=True)
 df.drop_duplicates(subset="id", inplace=True)
 df.reset_index(drop=True, inplace=True)

 df["soup"] = df.apply(make_soup, axis=1)
 vec = CountVectorizer(stop_words="english")
 sim = cosine_similarity(vec.fit_transform(df["soup"]))

 with open("similarity_matrix.pkl","wb") as f:
 pickle.dump(sim, f)
 with open("movies_df.pkl","wb") as f:
 pickle.dump(df, f)

 return df, sim


def recommend(title, df, sim, genre=None, n=10):
 idx_map = pd.Series(df.index, index=df["title"].str.lower())
 key = title.lower().strip()

 if key not in idx_map:
 matches = df[df["title"].str.lower().str.contains(key, na=False)]
 if matches.empty:
 return None, None
 idx = matches.index[0]
 matched = matches.iloc[0]["title"]
 else:
 idx = idx_map[key]
 matched = title
 if isinstance(idx, pd.Series):
 idx = idx.iloc[0]

 scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1:]
 results = []

 for i, score in scores:
 row = df.iloc[i]
 genres = get_genres(row["genres"])

 if genre and genre !="All" and genre not in genres:
 continue

 results.append({
"id": row["id"],
"title": row["title"],
"genres": genres,
"match": f"{round(score * 100, 1)}%",
"overview": str(row["overview"])[:110] +"..."
 })

 if len(results) >= n:
 break

 return results, matched


# page layout 

st.markdown('<p class="big-title"> Movie Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Type a movie you like and we\'ll find similar ones for you</p>', unsafe_allow_html=True)
st.markdown("---")

try:
 df, sim = load_model()
except FileNotFoundError:
 st.error("Dataset not found — make sure tmdb_5000_movies.csv and tmdb_5000_credits.csv are in the same folder as app.py")
 st.stop()

# collect all unique genres for the filter
all_genres = sorted(set(g for genres in df["genres"] for g in get_genres(genres)))

with st.sidebar:
 st.markdown("### Filters")
 genre_pick = st.selectbox("Genre", ["All"] + all_genres)
 how_many = st.slider("How many results", 5, 20, 10)
 st.markdown("---")
 st.caption(f"Dataset: {len(df):,} movies")
 st.caption("Method: Content-based filtering")

col1, col2 = st.columns([4, 1])
with col1:
 query = st.text_input("", placeholder="Search a movie... e.g. Inception, Avatar, Interstellar")
with col2:
 st.markdown("<br>", unsafe_allow_html=True)
 go = st.button("Search")

if go and query:
 results, matched = recommend(query, df, sim, genre_pick, how_many)

 if not results:
 st.error(f"Couldn't find '{query}' in the dataset. Try a different spelling.")
 else:
 if matched.lower() != query.lower():
 st.info(f"Showing results for: **{matched}**")

 st.markdown(f"### Results for **{matched}**")
 st.markdown("---")

 # show 5 cards per row
 for row_start in range(0, len(results), 5):
 chunk = results[row_start: row_start + 5]
 cols = st.columns(5)

 for col, movie in zip(cols, chunk):
 with col:
 poster = get_poster(movie["id"])
 genre_tags ="".join(f'<span class="gtag">{g}</span>' for g in movie["genres"][:3])
 st.markdown(f"""
 <div class="card">
 <img src="{poster}" alt="poster"/>
 <div class="name">{movie['title']}</div>
 <span class="badge">{movie['match']} match</span>
 <div style="margin-top:5px">{genre_tags}</div>
 </div>
""", unsafe_allow_html=True)

elif go and not query:
 st.warning("Please type a movie name first!")

else:
 st.markdown("##### Try searching for one of these:")
 picks = ["The Dark Knight","Inception","Interstellar","The Avengers","Avatar"]
 cols = st.columns(5)
 for col, title in zip(cols, picks):
 col.markdown(f" **{title}**")
 st.caption("Type any of the above in the search bar to get started.")
