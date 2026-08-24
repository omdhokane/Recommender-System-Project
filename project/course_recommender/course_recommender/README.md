# Coursewise — Student Course Recommendation System

A content-based course recommendation engine with a Flask web UI. Students
describe their interests/goals (and optionally a category or level), and the
system ranks the best-matching courses from the catalog using **TF-IDF +
cosine similarity**, with a small popularity boost from ratings/reviews.

## Project structure

```
course_recommender/
├── app.py                # Flask web app (routes + views)
├── recommender.py        # Core recommendation engine (also runnable as CLI)
├── requirements.txt
├── data/
│   └── courses.csv       # Sample dataset of 25 courses
├── templates/
│   ├── base.html
│   ├── index.html        # Search form + recommendation results
│   ├── similar.html      # "More like this course" page
│   └── browse.html       # Full catalog table
└── static/
    └── style.css
```

## How the recommendation logic works

1. Each course's **category, skills, description, and level** are combined
   into a single text "profile" (category and skills are weighted higher
   since they're the strongest relevance signals).
2. All course profiles are vectorized with **TF-IDF** (`scikit-learn`).
3. The student's free-text interests (+ optional category/level) are
   vectorized the same way and compared to every course via **cosine
   similarity**.
4. Scores are blended with a small **popularity factor** (`rating` and
   `log(num_reviews)`), so that among similarly relevant courses, better
   reviewed ones edge ahead.
5. Optional category/level filters narrow the candidate pool before ranking.

There's also a `similar_courses(course_id)` method for "if you liked X,
try these" style recommendations — useful once a student has completed
or rated a course.

## Setup

```bash
cd course_recommender
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- `/` — enter your interests, get ranked recommendations
- `/similar/<course_id>` — see courses similar to a given course (e.g. `/similar/C003`)
- `/browse` — view the full course catalog

## Run the CLI demo (no Flask needed)

```bash
python recommender.py
```

Answers a couple of prompts in the terminal and prints ranked recommendations.

## Extending this project

- **Swap in a real dataset**: replace `data/courses.csv` with a larger export
  from Coursera/Udemy/your institution's LMS — just keep the same columns
  (or update `_build_profiles()` in `recommender.py` to match new columns).
- **Add collaborative filtering**: once you have real student enrollment/
  rating history, blend in a user-item matrix (e.g. with `surprise` or
  matrix factorization) alongside the content-based score.
- **Add a database**: swap the CSV for SQLite/PostgreSQL and use SQLAlchemy
  if you want persistent student profiles, saved courses, or auth.
- **Deploy**: this is a standard Flask app — deployable as-is to Render,
  Railway, PythonAnywhere, or behind gunicorn + nginx.

## Notes for coursework / viva

If this is for an academic submission, you can point to:
- **Technique used**: Content-based filtering via TF-IDF vectorization and
  cosine similarity (`scikit-learn`).
- **Why content-based (not collaborative)**: no user rating history is
  required to cold-start recommendations — works from day one for any
  student.
- **Dataset**: `data/courses.csv`, 25 sample courses across 12 categories
  (swap in a larger real dataset for a production-grade demo).
