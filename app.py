import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 CareerCompass AI")
st.subheader("AI-Powered Career Recommendation System")

st.write("Discover your career path with AI.")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("CAREER PATH/outputs/cleaned_dataset.csv")

df = load_data()
df["Required_Skills"] = df["Required_Skills"].fillna("")

# User Input
st.header("👩‍💻 Enter Your Skills")

user_skills = st.text_input(
    "Enter skills separated by commas",
    placeholder="Python, SQL, Machine Learning"
)

if st.button("Analyze My Career"):

    if user_skills.strip():

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(
            df["Required_Skills"]
        )

        user_vector = vectorizer.transform([user_skills])

        similarity = cosine_similarity(
            user_vector, tfidf_matrix
        ).flatten()

        top_jobs = similarity.argsort()[-5:][::-1]

        st.success("Career Analysis Completed!")

        st.header("🎯 Top 5 Job Recommendations")

        for index in top_jobs:

            job = df.iloc[index]

            st.subheader(job["Job_Title"])

            st.write("Company:", job["Company"])

            st.write(
                "Match Score:",
                round(similarity[index] * 100, 2), "%"
            )

            st.write(
                "Required Skills:",
                job["Required_Skills"]
            )

            st.divider()

    else:
        st.warning("Please enter your skills.")
        
