import pickle
import streamlit as st
import pandas as pd

@st.cache_resource
def load_models():
    with open("models/similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    movies_df = pd.read_pickle("models/movies.pkl")
    return similarity, movies_df

similarity, movies_df = load_models()

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender")

movie_input = st.text_input(
    "Type a movie title and press Enter:",
    placeholder="e.g. The Dark Knight"
)

if st.button("Recommend"):
    # find exact match (you could also offer fuzzy ⚡ like before)
    matches = movies_df[movies_df['title'].str.lower() == movie_input.lower()]
    if matches.empty:
        st.error("No exact match found. Check spelling or try a different title.")
    else:
        idx = matches.index[0]
        # get list of (movie_idx, score), sorted descending
        sims = list(enumerate(similarity[idx]))
        sims.sort(key=lambda x: x[1], reverse=True)
        # skip first (itself), take next 5
        top5 = sims[1:6]

        st.subheader(f"Top 5 recommendations for “{movies_df.loc[idx,'title']}”")
        for i, score in top5:
            title = movies_df.loc[i, "title"]
            extra = ""
            if "director" in movies_df.columns:
                extra = f"— Directed by {movies_df.loc[i,'director']}"
            st.write(f"• **{title}** {extra}")
