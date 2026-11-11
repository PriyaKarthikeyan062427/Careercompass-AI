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
    return pd.read_csv("cleaned_dataset.csv")

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

        user_skill_list = [
            skill.strip().lower()
            for skill in user_skills.split(",")
            if skill.strip()
        ]

        # AI Recommendation Model
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

        # 1. Job Recommendations
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

        # Best matching job
        best_job = df.iloc[top_jobs[0]]

        # 2. Skill Gap Analysis
        st.header("📊 Skill Gap Analysis")

        required_skills = [
            skill.strip().lower()
            for skill in best_job["Required_Skills"].split(",")
            if skill.strip()
        ]

        missing_skills = [
            skill for skill in required_skills
            if skill not in user_skill_list
        ]

        st.write("Your Skills:", user_skills)
        st.write("Recommended Job:", best_job["Job_Title"])

        if missing_skills:
            st.warning("Skills You Need to Learn:")
            for skill in missing_skills:
                st.write("🔸", skill.title())
        else:
            st.success("Great! No skill gaps found.")

        # 3. Personalized Learning Roadmap
        st.header("📚 Personalized Learning Roadmap")

        learning_roadmap = {
            "python": "Learn Python basics, functions and Pandas.",
            "sql": "Learn SQL queries, joins and subqueries.",
            "machine learning": "Learn regression, classification and clustering.",
            "statistics": "Learn mean, median, probability and hypothesis testing.",
            "tensorflow": "Learn neural networks and TensorFlow basics.",
            "git": "Learn Git commands and GitHub.",
            "power bi": "Learn dashboards, charts and DAX.",
            "excel": "Learn formulas, pivot tables and data cleaning.",
            "data visualization": "Learn Matplotlib, Seaborn and Power BI.",
            "django": "Learn Django basics and web development."
        }

        if missing_skills:
            for skill in missing_skills:
                st.subheader(skill.title())
                st.write(
                    learning_roadmap.get(
                        skill,
                        "Search for beginner courses and practise with a small project."
                    )
                )
        else:
            st.success("Keep improving your existing skills!")

        # 4. Salary Analysis
        st.header("💰 Salary Analysis")

        salary_data = df.iloc[top_jobs].copy()

        salary_data["Min_Salary_LPA"] = pd.to_numeric(
            salary_data["Min_Salary_LPA"], errors="coerce"
        )
        salary_data["Max_Salary_LPA"] = pd.to_numeric(
            salary_data["Max_Salary_LPA"], errors="coerce"
        )

        st.dataframe(
            salary_data[
                ["Job_Title", "Company",
                 "Min_Salary_LPA", "Max_Salary_LPA"]
            ],
            use_container_width=True
        )

        average_salary = (
            salary_data["Min_Salary_LPA"].mean()
            + salary_data["Max_Salary_LPA"].mean()
        ) / 2

        st.metric(
            "Average Salary of Recommended Jobs",
            f"{average_salary:.2f} LPA"
        )

        # 5. Match Score Visualization
        st.header("📈 Job Match Score Visualization")

        chart_data = pd.DataFrame({
            "Job Role": df.iloc[top_jobs]["Job_Title"].values,
            "Match Score": (
                similarity[top_jobs] * 100
            ).round(2)
        })

        st.bar_chart(
            chart_data.set_index("Job Role")
        )

    else:
        st.warning("Please enter your skills.")
        
