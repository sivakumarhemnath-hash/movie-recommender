import pandas as pd
import numpy as np
import ast
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#configuration

MATRIX_FILE = "similarity_matrix.pkl"
DF_FILE     = "movies_df.pkl"

MOVIES_CSV  = r"C:\Users\acer\OneDrive\Documents\TMDB Dataset\tmdb_5000_movies.csv"
CREDITS_CSV = r"C:\Users\acer\OneDrive\Documents\TMDB Dataset\tmdb_5000_credits.csv"

#load data

def load_data():
    print(" Loading dataset...")

    if not os.path.exists(MOVIES_CSV) or not os.path.exists(CREDITS_CSV):
        print(" Dataset files not found! Check the file paths in CONFIGURATION.")
        exit()

    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    if "movie_id" in credits.columns:
        credits.rename(columns={"movie_id": "id"}, inplace=True)

    df = movies.merge(credits, on="id")
    df = df[["id", "title_x", "overview", "genres", "keywords", "cast", "crew"]]
    df.rename(columns={"title_x": "title"}, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset="id", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f" {len(df)} movies loaded.")
    return df

#feature extraction

def parse_list_field(field, key="name", limit=None):
    try:
        items = ast.literal_eval(str(field))
        names = [item[key].replace(" ", "") for item in items]
        return names[:limit] if limit else names
    except:
        return []


def get_director(crew_field):
    try:
        crew = ast.literal_eval(str(crew_field))
        return [m["name"].replace(" ", "") for m in crew if m.get("job") == "Director"]
    except:
        return []


def build_soup(row):
    genres   = parse_list_field(row["genres"])
    keywords = parse_list_field(row["keywords"])
    cast     = parse_list_field(row["cast"], limit=5)
    director = get_director(row["crew"])
    overview = str(row["overview"]).lower().split()
    return " ".join(director * 3 + genres * 2 + cast + keywords + overview)

#build or load similarity matrix

def get_similarity_matrix(df):
    if os.path.exists(MATRIX_FILE) and os.path.exists(DF_FILE):
        print(" Loading saved similarity matrix (instant!)...")
        with open(MATRIX_FILE, "rb") as f:
            similarity = pickle.load(f)
        with open(DF_FILE, "rb") as f:
            df = pickle.load(f)
        print(" Ready!")
        return df, similarity

    print("\n⚙  Building similarity matrix...")
    df["soup"] = df.apply(build_soup, axis=1)
    vectorizer = CountVectorizer(stop_words="english")
    matrix     = vectorizer.fit_transform(df["soup"])
    similarity = cosine_similarity(matrix, matrix)

    print(" Saving for instant future runs...")
    with open(MATRIX_FILE, "wb") as f:
        pickle.dump(similarity, f)
    with open(DF_FILE, "wb") as f:
        pickle.dump(df, f)

    print(" Done!")
    return df, similarity

#recommendation

def recommend(title, df, similarity, top_n=10):
    title_lower = title.lower().strip()
    indices     = pd.Series(df.index, index=df["title"].str.lower())

    if title_lower not in indices:
        close = df[df["title"].str.lower().str.contains(title_lower, na=False)]
        if close.empty:
            print(f" Movie '{title}' not found.")
            return pd.DataFrame()
        matched = close.iloc[0]["title"]
        print(f"  Exact match not found. Using closest: '{matched}'")
        idx = close.index[0]
    else:
        idx = indices[title_lower]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

    sim_scores    = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    sim_scores    = sim_scores[1: top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    scores        = [str(round(i[1] * 100, 2)) + "%" for i in sim_scores]

    result = df.iloc[movie_indices][["title"]].copy()
    result["match_%"] = scores
    result.reset_index(drop=True, inplace=True)
    result.index += 1
    return result

#main

def main():
    print(" Movie Recommendation System")
    print("=" * 50)

    df             = load_data()
    df, similarity = get_similarity_matrix(df)

    print("\n Commands:")
    print("   • Type any movie name → get 10 recommendations")
    print("   • 'quit'              → exit\n")

    while True:
        user_input = input(" Movie title: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print(" Goodbye!")
            break

        else:
            results = recommend(user_input, df, similarity, top_n=10)
            if not results.empty:
                print(f"\n Recommendations for '{user_input}':\n")
                print(results.to_string())
                print()


if __name__ == "__main__":
    main()
