import pickle
import streamlit as st
import pandas as pd

# ——— Load your two pickles once on startup ———
@st.cache_resource
def load_models():
    with open("models/similarity.pkl", "rb") as f:
        rec_model = pickle.load(f)
    # This should be the DataFrame you pickled (new_df with columns movie_id, title, tags, plus any others you need)
    movies_df = pd.read_pickle("models/movies.pkl")
    return rec_model, movies_df

similarity, movies_df = load_models()

# ——— Streamlit UI ———
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender")

# 1) Let user type any movie title
movie_input = st.text_input(
    "Type a movie title and press Enter:",
    placeholder="e.g. The Dark Knight"
)

if movie_input:
    # 2) Try to find it in your DataFrame
    matches = movies_df[movies_df['title'].str.lower().str.contains(movie_input.lower())]
    if matches.empty:
        st.error("No matching movies found. Check your spelling or try another title.")
    else:
        # If multiple partial matches, show them in a selectbox
        if len(matches) > 1:
            choice = st.selectbox(
                "Did you mean…",
                options=matches['title'].tolist()
            )
        else:
            choice = matches.iloc[0]['title']
        
        if st.button("Recommend"):
            try:
                idx = movies_df[movies_df['title'] == choice].index[0]
                dists, nbrs = similarity.kneighbors(
                    [movies_df.loc[idx, 'tags_vector']],  # or replace with your feature column
                    n_neighbors=6
                )
                rec_ids = nbrs[0][1:]  # skip the queried movie
                st.subheader(f"Recommendations based on “{choice}”")
                for i in rec_ids:
                    title = movies_df.loc[i, "title"]
                    info = ""
                    if "director" in movies_df.columns:
                        info = f"— Directed by {movies_df.loc[i, 'director']}"
                    st.write(f"• **{title}** {info}")
            except Exception:
                st.error("Oops, something went wrong generating recommendations.")
