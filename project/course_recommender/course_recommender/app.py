"""
app.py
------
Flask web application for the Student Course Recommendation System.
Wraps the CourseRecommender engine (recommender.py) with a simple UI.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
from recommender import CourseRecommender
import os

app = Flask(__name__)

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "courses.csv")
recommender = CourseRecommender(CSV_PATH)

CATEGORIES = sorted(recommender.df["category"].unique().tolist())
LEVELS = ["Beginner", "Intermediate", "Advanced"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", categories=CATEGORIES, levels=LEVELS)


@app.route("/recommend", methods=["POST"])
def recommend():
    interests = request.form.get("interests", "").strip()
    category = request.form.get("category", "").strip() or None
    level = request.form.get("level", "").strip() or None
    top_n = int(request.form.get("top_n", 6))

    error = None
    results = []

    if not interests:
        error = "Please tell us a bit about your interests or goals first."
    else:
        recs = recommender.recommend(
            interests=interests,
            preferred_category=category,
            preferred_level=level,
            top_n=top_n,
        )
        results = recs.to_dict(orient="records")

    return render_template(
        "index.html",
        categories=CATEGORIES,
        levels=LEVELS,
        results=results,
        query_interests=interests,
        query_category=category,
        query_level=level,
        error=error,
    )


@app.route("/similar/<course_id>")
def similar(course_id):
    try:
        course = recommender.df[recommender.df["course_id"] == course_id].iloc[0]
        sims = recommender.similar_courses(course_id, top_n=6).to_dict(orient="records")
        error = None
    except (IndexError, ValueError):
        course = None
        sims = []
        error = f"Course '{course_id}' not found."

    return render_template(
        "similar.html",
        course=course,
        results=sims,
        error=error,
    )


@app.route("/browse")
def browse():
    all_courses = recommender.df.to_dict(orient="records")
    return render_template("browse.html", courses=all_courses)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
