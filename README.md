# Intern ↔ Post Matcher (Admin Tool)

A web application built with **Flask** that helps administrators quickly identify which interns are the best fit for internship posts.  
The app uses **MiniLM sentence embeddings** and **cosine similarity** to match interns’ skills against post requirements, and ranks candidates by matchability.

## 🚀 Features
- Upload **CSV files** of interns and internship posts.
- Matches interns to posts based on **skill similarity** (using `all-MiniLM-L6-v2` model).
- Adjustable parameters:
  - **Top K per post** → show the top N interns for each internship.
  - **Minimum score** → filter out weak matches below a threshold.
- Polished **Bootstrap UI**:
  - Clean upload page
  - Dashboard-style results page
  - Loading screen with animated *“Matching the interns…”*
- Results displayed in **descending order of relevance**.

## 📂 Project Structure
intern_matcher/
│── app.py # Flask backend & frontend templates
│── requirements.txt # Python dependencies
│── data/
│ ├── interns.csv # Example interns dataset
│ ├── posts.csv # Example posts dataset

## 🛠️ Setup
1. Clone or download the repo.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Windows: venv\Scripts\activate)
