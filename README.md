# 🎬 MovieLens 100K Recommendation System

This project implements a **recommendation system** on the **MovieLens 100K dataset** using:

* **User-based Collaborative Filtering**
* **Item-based Collaborative Filtering**
* **Matrix Factorization (MF)
---

## 📂 Dataset

The project uses the **MovieLens 100K** dataset:

* 100,000 ratings from 943 users on 1,682 movies.
* Each user has rated at least 20 movies.

Files include:

* `u.data` → user_id, movie_id, rating, timestamp
* `u.item` → movie_id, title, genres
* `u1.base`, `u1.test` … `u5.base`, `u5.test` → train/test splits for cross-validation

Dataset link: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)

---

## 🛠 Methods

### 1️⃣ Collaborative Filtering (Memory-Based)

* **User-User Similarity** (cosine similarity between users)
* **Item-Item Similarity** (cosine similarity between movies)
* Predictions are computed using weighted averages of neighbors.

### 2️⃣ Matrix Factorization (Model-Based)

* Uses `cmfrec` to learn latent features for users & movies.
* Predicts ratings as:

[
\hat{r}_{u,i} = b + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i
]

* Can incorporate **side information** (e.g., genres).

---

## 📊 Evaluation

* **RMSE (Root Mean Squared Error)** is used to measure accuracy.
* Example:

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = test["rating"].values
y_pred = model.predict(test["user_id"].values, test["movie_id"].values)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)
```

---

## 🎯 Example Recommendations

```python
# Top-5 movie recommendations for User 1
recommend_ids = model.topN(
    user=1,
    n=5,
    exclude=train[train.user_id == 1].movie_id.values
)

# Convert IDs to titles
recommend_titles = [movie_dict[mid] for mid in recommend_ids]
print(recommend_titles)
```

Output:

```
['Return of the Jedi (1983)',
 'Star Wars (1977)',
 'Raiders of the Lost Ark (1981)',
 'Empire Strikes Back, The (1980)',
 'Indiana Jones and the Last Crusade (1989)']
```

---


## 📑 Requirements

* Python 3.8+
* pandas
* numpy
* scikit-learn
* cmfrec

---
