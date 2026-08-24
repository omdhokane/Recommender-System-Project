"""
recommender.py
----------------
Core recommendation logic for the Student Course Recommendation System.

Approach: Content-Based Filtering
    - Each course has a text profile built from its category, description,
      skills and level.
    - We vectorize all course profiles using TF-IDF.
    - A student's interests/skills (entered as free text, plus optional
      preferred category & level) are vectorized the same way.
    - Cosine similarity between the student's profile and every course
      profile gives a ranked list of the best-matching courses.

    A simple popularity boost (rating * log(num_reviews)) is blended in
    so that, among similarly relevant courses, better-reviewed ones rank
    slightly higher.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CourseRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self._build_profiles()
        self._fit_vectorizer()

    def _build_profiles(self):
        """Combine relevant text columns into a single 'profile' string per course."""
        self.df["profile"] = (
            (self.df["category"] + " ") * 3       # category weighted higher
            + (self.df["skills"] + " ") * 2         # skills weighted higher
            + self.df["description"] + " "
            + self.df["level"]
        ).str.lower()

    def _fit_vectorizer(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.course_matrix = self.vectorizer.fit_transform(self.df["profile"])

    @staticmethod
    def _popularity_score(rating: float, num_reviews: int) -> float:
        """Small boost for well-rated, well-reviewed courses (0-1 normalized-ish)."""
        return (rating / 5.0) * (np.log1p(num_reviews) / np.log1p(5000))

    def recommend(
        self,
        interests: str,
        preferred_category: str | None = None,
        preferred_level: str | None = None,
        top_n: int = 5,
        content_weight: float = 0.85,
        popularity_weight: float = 0.15,
    ) -> pd.DataFrame:
        """
        Return the top_n recommended courses for a student.

        Parameters
        ----------
        interests : free text describing interests, skills the student wants
                    to learn, or career goals. e.g. "I love python and want
                    to get into machine learning and AI"
        preferred_category : optional exact/partial category filter, e.g. "Data Science"
        preferred_level : optional level filter: Beginner / Intermediate / Advanced
        top_n : number of results to return
        """
        query = interests.lower()
        if preferred_category:
            query += f" {preferred_category.lower()} " * 2
        if preferred_level:
            query += f" {preferred_level.lower()}"

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.course_matrix).flatten()

        pop_scores = self.df.apply(
            lambda r: self._popularity_score(r["rating"], r["num_reviews"]), axis=1
        ).values

        final_score = content_weight * sims + popularity_weight * pop_scores

        result = self.df.copy()
        result["match_score"] = np.round(final_score * 100, 1)
        result["similarity"] = np.round(sims * 100, 1)

        if preferred_category:
            mask = result["category"].str.lower().str.contains(
                preferred_category.lower()
            )
            if mask.any():
                result = result[mask]

        if preferred_level:
            mask = result["level"].str.lower() == preferred_level.lower()
            if mask.any():
                result = result[mask]

        result = result.sort_values("match_score", ascending=False).head(top_n)
        return result[
            [
                "course_id",
                "course_name",
                "category",
                "instructor",
                "level",
                "duration_weeks",
                "rating",
                "num_reviews",
                "match_score",
                "description",
                "skills",
            ]
        ].reset_index(drop=True)

    def similar_courses(self, course_id: str, top_n: int = 5) -> pd.DataFrame:
        """Given a course the student already likes/completed, find similar ones."""
        if course_id not in self.df["course_id"].values:
            raise ValueError(f"Course id '{course_id}' not found.")

        idx = self.df.index[self.df["course_id"] == course_id][0]
        sims = cosine_similarity(
            self.course_matrix[idx], self.course_matrix
        ).flatten()

        result = self.df.copy()
        result["match_score"] = np.round(sims * 100, 1)
        result = result[result["course_id"] != course_id]
        result = result.sort_values("match_score", ascending=False).head(top_n)
        return result[
            ["course_id", "course_name", "category", "level", "rating", "match_score"]
        ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Simple command-line demo (run: python recommender.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rec = CourseRecommender("data/courses.csv")

    print("=" * 70)
    print("STUDENT COURSE RECOMMENDATION SYSTEM - CLI DEMO")
    print("=" * 70)

    interests = input(
        "\nDescribe your interests / skills you want to learn:\n> "
    ).strip() or "I want to learn python and get into data science and machine learning"

    category = input(
        "\nPreferred category (press Enter to skip): "
    ).strip() or None

    level = input(
        "Preferred level - Beginner/Intermediate/Advanced (press Enter to skip): "
    ).strip() or None

    recommendations = rec.recommend(interests, category, level, top_n=5)

    print("\nTop Recommended Courses:\n")
    for i, row in recommendations.iterrows():
        print(f"{i + 1}. {row['course_name']}  ({row['category']} | {row['level']})")
        print(f"   Instructor: {row['instructor']}  |  Rating: {row['rating']}⭐ "
              f"({row['num_reviews']} reviews)  |  Duration: {row['duration_weeks']} weeks")
        print(f"   Match Score: {row['match_score']}%")
        print(f"   {row['description']}")
        print("-" * 70)
