import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Medicine Assistant",
    page_icon="💊",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/medicines_cleaned.csv"
    )

    embeddings = np.load(
        "models/medicine_embeddings.npy"
    )

    return df, embeddings


# --------------------------------------------------
# LOAD TRANSFORMER
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


df, embeddings = load_data()
model = load_model()


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def search_medicine(query, top_k=5):

    query_embedding = model.encode(
        [query]
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()

    results["Similarity"] = similarities[top_indices]

    return results


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("💊 AI Medicine Information System")

st.write(
    "Search for a medicine or describe what you are looking for."
)

st.info(
    "This application provides informational results from the "
    "dataset and is not a substitute for a doctor or pharmacist."
)


# --------------------------------------------------
# SEARCH BOX
# --------------------------------------------------

query = st.text_input(
    "🔍 Search Medicine",
    placeholder="Example: Dolo 650 or medicine for bacterial infection"
)


search_button = st.button(
    "Search",
    type="primary"
)


# --------------------------------------------------
# SEARCH RESULTS
# --------------------------------------------------

if search_button and query.strip():

    results = search_medicine(
        query,
        top_k=5
    )

    best = results.iloc[0]

    st.markdown("---")

    # --------------------------------------------------
    # BEST MATCH
    # --------------------------------------------------

    st.subheader("🎯 Best Match")

    col1, col2 = st.columns([1, 2])

    with col1:

        try:

            st.image(
                best["Image URL"],
                caption=best["Medicine Name"],
                width="stretch"
            )

        except Exception:

            st.warning(
                "Medicine image could not be loaded."
            )


    with col2:

        st.header(
            best["Medicine Name"]
        )

        similarity = best["Similarity"]

        st.metric(
            "Semantic Similarity",
            f"{similarity:.2%}"
        )

        st.markdown("### 🧪 Composition")

        st.write(
            best["Composition"]
        )

        st.markdown("### 📋 Uses")

        st.write(
            best["Uses"]
        )

        st.markdown("### ⚠️ Side Effects")

        st.write(
            best["Side_effects"]
        )

        st.markdown("### 🏭 Manufacturer")

        st.write(
            best["Manufacturer"]
        )


    # --------------------------------------------------
    # REVIEWS
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("⭐ Review Distribution")

    review_col1, review_col2, review_col3 = st.columns(3)

    with review_col1:

        st.metric(
            "Excellent",
            f"{best['Excellent Review %']}%"
        )

    with review_col2:

        st.metric(
            "Average",
            f"{best['Average Review %']}%"
        )

    with review_col3:

        st.metric(
            "Poor",
            f"{best['Poor Review %']}%"
        )


    # --------------------------------------------------
    # SIMILAR MEDICINES
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("🔎 Similar Medicines")

    similar = results.iloc[1:]

    for _, medicine in similar.iterrows():

        with st.container():

            col1, col2 = st.columns([1, 4])

            with col1:

                try:

                    st.image(
                        medicine["Image URL"],
                        width=120
                    )

                except Exception:

                    pass

            with col2:

                st.markdown(
                    f"### {medicine['Medicine Name']}"
                )

                st.write(
                    medicine["Composition"]
                )

                st.write(
                    f"Similarity: "
                    f"{medicine['Similarity']:.2%}"
                )

                st.markdown("---")


elif search_button:

    st.warning(
        "Please enter a medicine name or search query."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "AI Medicine Information System | "
    "Transformer-based semantic retrieval"
)