# talentai/grading.py
from __future__ import annotations
import ast, textwrap
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class GradeResult:
    score: int
    max_score: int
    feedback: str
    details: Dict[str, Any]

# Challenge A: "Model Debug Sprint" (classification on a toy dataset)
BUGGY_SNIPPET = """
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LinearRegression()
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)           # <- not integer labels!
acc = accuracy_score(yte, pred)   # wrong metric with floats
print(acc)
"""

# Expected fixes (heuristic checks): use a classifier, discrete preds, accuracy_score on labels.
KEY_REQS = ["LogisticRegression", "KNeighborsClassifier", "SVC", "DecisionTreeClassifier",
            "RandomForestClassifier"]

def grade_debug_submission(user_code: str) -> GradeResult:
    code = textwrap.dedent(user_code)
    tree = ast.parse(code)
    src = code

    points = 0; max_points = 4
    # 1) model is classifier
    if any(k in src for k in KEY_REQS): points += 1
    # 2) accuracy_score used on integer labels (naive: user casts with ast.Name 'predict' then argmax/round)
    if "accuracy_score" in src: points += 1
    # 3) Predict → class labels (heuristic)
    if ".predict(" in src and (".round(" in src or "astype(int)" in src or "predict(" in src and "KNeighborsClassifier" in src):
        points += 1
    # 4) No LinearRegression usage
    if "LinearRegression" not in src: points += 1

    fb = []
    if points < max_points:
        fb.append("Hints: switch to a classifier (e.g., LogisticRegression), ensure predictions are class labels, and compute accuracy on labels.")
    return GradeResult(points, max_points, "\n".join(fb), {"checks": points})

# Challenge B: "Model ↔ Plot" matching
PLOTS = {
    "plot_1.png": "SVM (RBF)",
    "plot_2.png": "Decision Tree",
    "plot_3.png": "kNN",
    "plot_4.png": "Logistic Regression",
}
def grade_plot_matching(answers: Dict[str,str]) -> GradeResult:
    correct = sum(PLOTS[k] == v for k, v in answers.items())
    fb = f"You matched {correct}/{len(PLOTS)} correctly."
    return GradeResult(correct, len(PLOTS), fb, {"correct_map": PLOTS})
