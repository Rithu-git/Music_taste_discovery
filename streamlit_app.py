import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Music Taste Discovery", page_icon="🎵", layout="wide")

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]
SHOW = ["track_name", "artists", "track_genre", "popularity"]
K = 6


# ------------------------------------------------------------------
# Load data, preprocess, cluster, mine rules (runs once, then cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models... (first load takes 1-2 minutes)")
def build():
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
    else:
        df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")

    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["track_name", "artists"])
    df = df.drop_duplicates(subset="track_id")
    df = df.drop_duplicates(subset=["track_name", "artists"])
    df = df[df.tempo > 0]
    lo, hi = df[FEATURES].quantile(0.01), df[FEATURES].quantile(0.99)
    df[FEATURES] = df[FEATURES].clip(lower=lo, upper=hi, axis=1)
    df = df.reset_index(drop=True)

    X = StandardScaler().fit_transform(df[FEATURES])
    P = PCA(n_components=2, random_state=42).fit_transform(X)
    df["pc1"], df["pc2"] = P[:, 0], P[:, 1]

    df["cluster"] = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X)
    sil = silhouette_score(X, df.cluster, sample_size=10000, random_state=42)
    dbi = davies_bouldin_score(X, df.cluster)

    profile = pd.DataFrame(X, columns=FEATURES).groupby(df.cluster).mean()
    names = {}
    for k, row in profile.iterrows():
        top = row.sort_values(ascending=False).index[:2]
        names[k] = f"High {top[0]} / {top[1]}"

    # Association rules
    genres = ["pop", "rock", "hip-hop", "jazz", "classical", "edm", "metal", "country",
              "reggae", "blues", "techno", "latin", "acoustic", "dance"]
    sub = df[df.track_genre.isin(genres)].copy()
    bin_cols = ["energy", "danceability", "valence", "acousticness", "tempo", "loudness", "speechiness"]
    for c in bin_cols:
        sub[c + "_item"] = c + "=" + pd.qcut(sub[c].rank(method="first"), 3,
                                             labels=["Low", "Mid", "High"]).astype(str)
    sub["genre_item"] = "genre=" + sub.track_genre
    items = sub[[c + "_item" for c in bin_cols] + ["genre_item"]]
    trans = pd.get_dummies(items, prefix="", prefix_sep="", dtype=bool)
    freq = apriori(trans, min_support=0.03, use_colnames=True, max_len=3)
    try:
        rules = association_rules(freq, metric="lift", min_threshold=1.2)
    except TypeError:
        rules = association_rules(freq, num_itemsets=len(trans), metric="lift", min_threshold=1.2)
    gr = rules[rules.consequents.apply(lambda s: len(s) == 1 and list(s)[0].startswith("genre="))]
    gr = gr[gr.confidence >= 0.3].sort_values("lift", ascending=False)
    rt = gr[["antecedents", "consequents", "support", "confidence", "lift"]].head(20).copy()
    rt["antecedents"] = rt.antecedents.apply(lambda s: ", ".join(sorted(s)))
    rt["consequents"] = rt.consequents.apply(lambda s: ", ".join(sorted(s)))
    rt = rt.round(3).reset_index(drop=True)

    summary = pd.DataFrame({
        "cluster": list(names.keys()),
        "name": list(names.values()),
        "size": df.cluster.value_counts().sort_index().values,
        "top_genres": [", ".join(df[df.cluster == k].track_genre.value_counts().index[:3]) for k in names],
    })
    return df, X, names, sil, dbi, rt, summary


df, X, cluster_names, SIL, DBI, RULES, SUMMARY = build()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def find_track(q):
    m = df[df.track_name.str.lower().str.contains(q.strip().lower(), regex=False)]
    return None if m.empty else m.popularity.idxmax()


def recommend_core(seed_idx, n=10, genre_w=0.15, pop_w=0.10, max_per_artist=2):
    v = X[seed_idx].mean(axis=0, keepdims=True)
    sim = cosine_similarity(v, X)[0]

    # hybrid score: audio similarity + genre match + popularity
    seed_genres = set(df.track_genre.iloc[seed_idx])
    score = sim + genre_w * df.track_genre.isin(seed_genres).values + pop_w * (df.popularity.values / 100)

    seed_titles = set(df.track_name.iloc[seed_idx].str.lower())
    order = np.argsort(-score)
    top, artist_count = [], {}
    for i in order[:2000]:
        if i in seed_idx or df.track_name.iat[i].lower() in seed_titles:
            continue
        a = df.artists.iat[i]
        if artist_count.get(a, 0) >= max_per_artist:
            continue
        artist_count[a] = artist_count.get(a, 0) + 1
        top.append(i)
        if len(top) == n:
            break
    top = np.array(top)
    return top, sim[top]


def plot_map(rec_idx, seed_idx=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    s = df.sample(8000, random_state=1)
    ax.scatter(s.pc1, s.pc2, c=s.cluster, cmap="tab10", s=4, alpha=0.3)
    ax.scatter(df.loc[rec_idx, "pc1"], df.loc[rec_idx, "pc2"], c="red", s=50,
               edgecolor="k", label="Recommended")
    if seed_idx is not None:
        ax.scatter(df.loc[seed_idx, "pc1"], df.loc[seed_idx, "pc2"], marker="*", s=300,
                   c="gold", edgecolor="k", label="Your songs")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Music Map (PCA)")
    ax.legend()
    return fig


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------
st.title("🎵 Music Taste Discovery & Recommendation System")
st.caption(f"Data Mining Project: K-Means clustering, association rules and content-based "
           f"filtering on {len(df):,} Spotify tracks.")

tab1, tab2, tab3, tab4 = st.tabs(["🎧 Taste-based", "🎚️ Mood-based",
                                   "🗺️ Cluster explorer", "📊 Mining results"])

with tab1:
    st.write("**Try these popular songs:**")
    st.write(", ".join(df.nlargest(6, "popularity").track_name.tolist()))
    text = st.text_area("Favorite songs (one per line)", height=120)
    n1 = st.slider("Number of recommendations", 5, 20, 10, key="n1")
    if st.button("Recommend", type="primary"):
        seeds, missing = [], []
        for q in [l for l in text.split("\n") if l.strip()]:
            i = find_track(q)
            if i is not None:
                seeds.append(i)
            else:
                missing.append(q)
        if not seeds:
            st.error("No matching songs found. Try different titles.")
        else:
            top, sims = recommend_core(seeds, n1)
            out = df.loc[top, SHOW].copy()
            out["similarity"] = sims.round(3)
            main_cl = int(df.cluster[seeds].mode()[0])
            st.success("Matched: " + "; ".join(f"{df.track_name[i]} - {df.artists[i]}" for i in seeds))
            st.info(f"Your taste cluster: #{main_cl} ({cluster_names[main_cl]})")
            if missing:
                st.warning("Not found: " + ", ".join(missing))
            c1, c2 = st.columns([1, 1])
            c1.dataframe(out, use_container_width=True)
            c2.pyplot(plot_map(top, seeds))

with tab2:
    c1, c2 = st.columns(2)
    energy = c1.slider("Energy", 0.0, 1.0, 0.7)
    valence = c1.slider("Valence (happiness)", 0.0, 1.0, 0.6)
    dance = c2.slider("Danceability", 0.0, 1.0, 0.7)
    acoustic = c2.slider("Acousticness", 0.0, 1.0, 0.2)
    n2 = st.slider("Number of recommendations", 5, 20, 10, key="n2")
    target = {"energy": energy, "valence": valence, "danceability": dance, "acousticness": acoustic}
    d = sum((df[c] - v) ** 2 for c, v in target.items()) ** 0.5
    cand = d.nsmallest(100).index
    top = df.loc[cand].sort_values("popularity", ascending=False).head(n2).index
    a, b = st.columns(2)
    a.dataframe(df.loc[top, SHOW], use_container_width=True)
    b.pyplot(plot_map(top))

with tab3:
    choice = st.selectbox("Taste cluster", [f"{k}: {cluster_names[k]}" for k in sorted(cluster_names)])
    k = int(choice.split(":")[0])
    s = df[df.cluster == k].nlargest(20, "popularity")
    a, b = st.columns(2)
    a.dataframe(s[SHOW], use_container_width=True)
    b.pyplot(plot_map(s.index))

with tab4:
    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette score", f"{SIL:.3f}")
    m2.metric("Davies-Bouldin (lower = better)", f"{DBI:.3f}")
    m3.metric("Clusters (K)", K)
    st.subheader("Cluster summary")
    st.dataframe(SUMMARY, use_container_width=True)
    st.subheader("Association rules: audio profile → genre (sorted by lift)")
    st.dataframe(RULES, use_container_width=True)
