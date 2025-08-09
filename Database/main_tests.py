# main.py
from db_setup import init_db
from recruitment import create_user, login_user, add_job, add_candidate_details, get_all_jobs, get_candidates

# 1. Initialize the database
init_db()

# 2. Create a recruiter
create_user("recruiter@example.com", "StrongPass123", "recruiter")

# 3. Recruiter logs in
uid, role = login_user("recruiter@example.com", "StrongPass123")

# 4. Post a job
if uid and role == "recruiter":
    add_job(uid, "Data Science", "Senior Data Scientist", "Remote - US",
            "Build ML models", "Health insurance", "Python, SQL, ML",
            "Bachelor's in CS", "Senior", 5, "Full-time", "Urgent hire")

# 5. Create a candidate
create_user("candidate@example.com", "SecurePass456", "candidate")
uid_cand, role_cand = login_user("candidate@example.com", "SecurePass456")

if uid_cand and role_cand == "candidate":
    add_candidate_details(uid_cand, "Jane Doe", "Data Analyst",
                          "Passionate about analytics", "New York",
                          "Remote", 3, "Mid-level")

# 6. Print all jobs
print("\n📌 Jobs in the system:")
for job in get_all_jobs():
    print(job)

# 7. Print all candidates
print("\n📌 Candidates in the system:")
for cand in get_candidates():
    print(cand)
