import random
import time
import pandas as pd

DEBUG_SNIPPETS = {
    "mlops": [
        {
            "prompt": "Find the bug in this pipeline snippet (PyTorch training loop).",
            "code": "for epoch in range(epochs):\n  model.train()\n  for X, y in loader:\n    preds = model(X)\n    loss = criterion(preds, y)\n    loss.backward()\n    optimizer.step()\n    # BUG: missing optimizer.zero_grad()",
            "options": [
                "The optimizer is never zeroed; add optimizer.zero_grad() before backward.",
                "criterion should be called without preds.",
                "Use model.eval() in training.",
                "Data loader must be recreated in each epoch."
            ],
            "answer_idx": 0
        }
    ],
    "llmops": [
        {
            "prompt": "RAG pipeline throughput is low. Best first fix?",
            "code": "(No code)",
            "options": [
                "Increase chunk overlap to 80%",
                "Batch embedding calls / enable async I/O to the vector DB",
                "Use a larger LLM",
                "Store documents as images"
            ],
            "answer_idx": 1
        }
    ],
    "cv": [
        {
            "prompt": "Model underfits on CIFAR-10. What helps first?",
            "code": "(No code)",
            "options": [
                "Drastically reduce training data",
                "Remove all augmentation",
                "Increase capacity (e.g., WiderResNet) and train longer with LR schedule",
                "Disable normalization"
            ],
            "answer_idx": 2
        }
    ]
}

def _parse_skills_field(text):
    if not isinstance(text, str):
        return []
    # split on commas or slashes or pipes
    raw = [t.strip() for t in text.replace("/",",").replace("|",",").split(",")]
    return [r for r in raw if r]

def build_quiz(job_row: pd.Series, global_skill_pool: list) -> list[dict]:
    req_skills = _parse_skills_field(job_row.get("Skills/Tech-stack required",""))
    topic = str(job_row.get("topic","")).lower()
    area = "mlops" if "mlops" in topic else "llmops" if "llm" in topic else "cv" if "vision" in topic else "mlops"
    decoys = [s for s in global_skill_pool if s not in req_skills]
    random.shuffle(decoys)

    quiz = []

    # Q1: NOT required
    opts = (req_skills[:2] + decoys[:2])[:4]
    random.shuffle(opts)
    quiz.append({
        "type": "single",
        "question": "Which of the following is NOT required for this job?",
        "options": opts,
        "answer_idx": opts.index(next((o for o in opts if o not in req_skills), opts[0]))
    })

    # Q2: YOE
    y = str(job_row.get("yoe","")).strip() or "0"
    opts = [y, "1", "2", "3", "5", "7"]
    opts = list(dict.fromkeys(opts))[:4]  # unique, max 4
    random.shuffle(opts)
    quiz.append({
        "type": "single",
        "question": "What is the minimum years of experience (YOE) required?",
        "options": opts,
        "answer_idx": opts.index(str(y))
    })

    # Q3: employment type
    et = str(job_row.get("employment_type","")).strip() or "Full Time"
    opts = [et, "Contract", "Part Time", "Internship"]
    random.shuffle(opts)
    quiz.append({
        "type": "single",
        "question": "What is the employment type for this role?",
        "options": opts,
        "answer_idx": opts.index(et)
    })

    # Q4: central skills (multi)
    correct = req_skills[:2] if len(req_skills)>=2 else req_skills[:1]
    opts = list(dict.fromkeys((req_skills[:3] + decoys[:5])[:6]))
    random.shuffle(opts)
    quiz.append({
        "type": "multi",
        "question": "Select the TWO most central skills for this role:",
        "options": opts,
        "answer_set": set(correct[:2])
    })

    # Q5: tiny debug challenge
    dbg = random.choice(DEBUG_SNIPPETS[area])
    quiz.append({
        "type": "single",
        "question": dbg["prompt"],
        "code": dbg["code"],
        "options": dbg["options"],
        "answer_idx": dbg["answer_idx"]
    })

    return quiz

def grade_quiz(answers: list, quiz: list) -> dict:
    score = 0
    max_score = len(quiz)
    for ans, q in zip(answers, quiz):
        if q["type"] == "single":
            if ans is not None and int(ans) == int(q["answer_idx"]):
                score += 1
        elif q["type"] == "multi":
            sel = set(ans or [])
            if sel == q["answer_set"]:
                score += 1
    pct = round(100.0 * score / max_score, 1) if max_score else 0.0
    return {"score_raw": score, "score_max": max_score, "score_pct": pct, "timestamp": int(time.time())}
