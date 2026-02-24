import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training data
resumes = [
    "Python Machine Learning Data Science",
    "Java Developer Spring Boot Backend",
    "Deep Learning AI Python Tensorflow",
    "Accounting Finance Excel Tally"
]

labels = [1, 0, 1, 0]  # 1 = Selected, 0 = Not Selected

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resumes)

model = LogisticRegression()
model.fit(X, labels)

st.title("AI Resume Screening System")

job_desc = st.text_area("Enter Job Description")
resume_input = st.text_area("Paste Resume Content")

if st.button("Analyze Resume"):
    input_data = vectorizer.transform([resume_input])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Resume Selected for Interview")
    else:
        st.error("❌ Resume Not Suitable")
