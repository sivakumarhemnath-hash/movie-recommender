"""
Compress tmdb_5000_credits.csv
================================
Shrinks the credits file by keeping only top 5 cast and director.
Run this ONCE — it creates a smaller 'tmdb_5000_credits_small.csv'
that you can upload to GitHub easily.

Run:
    python compress_credits.py
"""

import pandas as pd
import ast
import os

CREDITS_CSV = r"C:\Users\acer\OneDrive\Documents\TMDB Dataset\tmdb_5000_credits.csv"
OUTPUT_CSV  = r"C:\Users\acer\OneDrive\Documents\TMDB Dataset\tmdb_5000_credits_small.csv"


def shrink_cast(cast_str):
    """Keep only top 5 cast names."""
    try:
        cast = ast.literal_eval(str(cast_str))
        top5 = [{"name": c["name"]} for c in cast[:5]]
        return str(top5)
    except:
        return "[]"


def shrink_crew(crew_str):
    """Keep only the director."""
    try:
        crew = ast.literal_eval(str(crew_str))
        directors = [{"job": "Director", "name": m["name"]}
                     for m in crew if m.get("job") == "Director"]
        return str(directors)
    except:
        return "[]"


print("Loading credits file...")
df = pd.read_csv(CREDITS_CSV)

print("Shrinking cast and crew columns...")
df["cast"] = df["cast"].apply(shrink_cast)
df["crew"] = df["crew"].apply(shrink_crew)

df.to_csv(OUTPUT_CSV, index=False)

original_size = os.path.getsize(CREDITS_CSV) / (1024 * 1024)
new_size      = os.path.getsize(OUTPUT_CSV)  / (1024 * 1024)

print(f"\nDone!")
print(f"  Original size : {original_size:.1f} MB")
print(f"  New size      : {new_size:.1f} MB")
print(f"  Saved to      : {OUTPUT_CSV}")
print(f"\nNow upload 'tmdb_5000_credits_small.csv' to GitHub instead!")
