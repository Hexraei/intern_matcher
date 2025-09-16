from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re

print("Loading MiniLM model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

app = Flask(__name__)

# ------------------ Helpers ------------------
def detect_column(df, candidates):
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for col in df.columns:
            if cand.lower() == col.lower():
                return col
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    return df.columns[0]

def split_skills(s):
    if pd.isna(s):
        return []
    s = str(s)
    s = re.sub(r"[;/|]+", ",", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seen, out = set(), []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def prepare_embeddings(df, skills_col):
    df = df.copy()
    df["skills_list"] = df[skills_col].apply(split_skills)
    df["skills_text"] = df["skills_list"].apply(lambda L: ", ".join(L) if L else "")
    texts = df["skills_text"].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return df, embeddings

# ------------------ HTML Templates ------------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Intern ↔ Post Matcher (Admin)</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #f4f6f9; }
    .upload-card { border-radius: 1rem; }
    .btn-primary { border-radius: 50px; padding: 0.6rem 1.4rem; }
    .form-control { border-radius: 0.5rem; }
    #loading-overlay {
      position: fixed; top:0; left:0; width:100%; height:100%;
      background: rgba(255,255,255,0.9); z-index:9999;
      display:none; align-items:center; justify-content:center; flex-direction:column;
    }
    .loading-text {
      font-size:1.5rem; font-weight:600; color:#333;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { opacity: 0.3; }
      50% { opacity: 1; }
      100% { opacity: 0.3; }
    }
  </style>
</head>
<body>
<div class="container py-5">
  <div class="row justify-content-center">
    <div class="col-lg-7">
      <div class="card shadow upload-card p-4">
        <div class="card-body text-center">
          <h2 class="mb-4">Intern ↔ Post Matcher (Admin)</h2>
          <form action="/match" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <div class="mb-4">
              <label class="form-label fw-bold">Interns CSV</label>
              <input class="form-control" type="file" name="interns" accept=".csv" required>
              <small class="text-muted">Must include a column for skills.</small>
            </div>
            <div class="mb-4">
              <label class="form-label fw-bold">Posts CSV</label>
              <input class="form-control" type="file" name="posts" accept=".csv" required>
              <small class="text-muted">Must include a column for required skills.</small>
            </div>
            <div class="row g-3 mb-4">
              <div class="col-md-6">
                <label class="form-label">Top K per post</label>
                <input class="form-control" type="number" name="top_k" value="10" min="1">
              </div>
              <div class="col-md-6">
                <label class="form-label">Minimum score (0.0 - 1.0)</label>
                <input class="form-control" type="number" step="0.01" min="0" max="1" name="min_score" value="0.00">
              </div>
            </div>
            <button class="btn btn-primary btn-lg" type="submit">⚡ Match Interns</button>
          </form>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Loading overlay -->
<div id="loading-overlay">
  <div class="spinner-border text-primary mb-3" style="width:3rem;height:3rem;" role="status"></div>
  <div class="loading-text">Matching the interns...</div>
</div>

<script>
function showLoading() {
  document.getElementById("loading-overlay").style.display = "flex";
}
</script>
</body>
</html>
"""

RESULTS_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Matches</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #f4f6f9; }
    .card { border-radius: 1rem; }
    .score { font-weight: 700; color: #0d6efd; }
  </style>
</head>
<body>
<div class="container py-4">
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h2 class="fw-bold">Matching Results</h2>
    <a class="btn btn-outline-secondary" href="/">← Upload New CSVs</a>
  </div>

  {% for r in results %}
    <div class="card mb-4 shadow-sm">
      <div class="card-body">
        <h5 class="card-title text-primary mb-3">📌 Post: {{ r.post_title }}</h5>
        {% if r.rows %}
          <div class="table-responsive">
            <table class="table table-hover align-middle">
              <thead class="table-light">
                <tr>
                  <th scope="col">Rank</th>
                  <th scope="col">Intern Name</th>
                  <th scope="col">Score</th>
                  <th scope="col">Skills</th>
                  <th scope="col">Other Info</th>
                </tr>
              </thead>
              <tbody>
                {% for row in r.rows %}
                  <tr>
                    <td>{{ loop.index }}</td>
                    <td><strong>{{ row.get(interns_name_col, row.get('name', '')) }}</strong></td>
                    <td class="score">{{ ('{:.2%}'.format(row['score'])) }}</td>
                    <td>{{ row.get('skills_text','') }}</td>
                    <td>
                      {% for k,v in row.items() %}
                        {% if k not in ['skills_text','skills_list','score'] %}
                          <small><strong>{{ k }}:</strong> {{ v }}</small><br>
                        {% endif %}
                      {% endfor %}
                    </td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% else %}
          <p class="text-muted">No candidates passed the minimum score.</p>
        {% endif %}
      </div>
    </div>
  {% endfor %}
</div>
</body>
</html>
"""

# ------------------ Routes ------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/match", methods=["POST"])
def match():
    interns_file = request.files.get("interns")
    posts_file = request.files.get("posts")
    if not interns_file or not posts_file:
        return "Both CSVs required.", 400

    interns_df = pd.read_csv(interns_file)
    posts_df = pd.read_csv(posts_file)

    interns_name_col = detect_column(interns_df, ["name","candidate"])
    interns_skills_col = detect_column(interns_df, ["skills","skillset"])
    posts_name_col = detect_column(posts_df, ["post","title","name"])
    posts_skills_col = detect_column(posts_df, ["skills","skillset"])

    interns_df, interns_embs = prepare_embeddings(interns_df, interns_skills_col)
    posts_df, posts_embs = prepare_embeddings(posts_df, posts_skills_col)

    top_k = int(request.form.get("top_k", 10))
    min_score = float(request.form.get("min_score", 0.0))

    results = []
    for i, post_row in posts_df.iterrows():
        sims = cosine_similarity(posts_embs[i].reshape(1, -1), interns_embs).flatten()
        tmp = interns_df.copy()
        tmp["score"] = sims
        tmp = tmp[tmp["score"] >= min_score].sort_values("score", ascending=False).head(top_k)
        rows = tmp.to_dict(orient="records")
        for rr in rows:
            for k, v in list(rr.items()):
                if isinstance(v, (np.floating, np.integer)):
                    rr[k] = float(v) if isinstance(v, np.floating) else int(v)
        results.append({
            "post_title": str(post_row.get(posts_name_col, f"Post {i}")),
            "rows": rows
        })

    return render_template_string(RESULTS_HTML, results=results, interns_name_col=interns_name_col)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
