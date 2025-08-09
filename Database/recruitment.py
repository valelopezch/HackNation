# recruitment.py
import sqlite3
import bcrypt
from datetime import datetime

DB_NAME = "recruitment.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

# ---------------- USERS ----------------
def create_user(email, password, role):
    conn = get_connection()
    cursor = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
                       (email, hashed, role))
        conn.commit()
        print("✅ User created successfully.")
    except sqlite3.IntegrityError:
        print("❌ Email already exists.")
    conn.close()

def login_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, password_hash, role FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode('utf-8'), result[1]):
        print(f"✅ Login successful. Role: {result[2]}")
        return result[0], result[2]
    else:
        print("❌ Invalid credentials.")
        return None, None

# ---------------- CANDIDATE DETAILS ----------------
def add_candidate_details(user_id, full_name, candidate_title, about,
                          location, preferred_employment_type, yoe, seniority):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO candidate_details (
            user_id, full_name, candidate_title, about, location,
            preferred_employment_type, yoe, seniority
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, full_name, candidate_title, about,
          location, preferred_employment_type, yoe, seniority))
    conn.commit()
    conn.close()
    print("✅ Candidate details added.")

# ---------------- JOBS ----------------
def add_job(posted_by, topic, job_title, site, tasks, perks_benefits,
            skills_tech_stack, educational_requirements, seniority, yoe,
            employment_type, extra_info):
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO jobs (posted_by, topic, job_title, site, tasks, perks_benefits,
                          skills_tech_stack, educational_requirements, seniority, yoe,
                          employment_type, extra_info, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (posted_by, topic, job_title, site, tasks, perks_benefits,
          skills_tech_stack, educational_requirements, seniority, yoe,
          employment_type, extra_info, created_at))
    conn.commit()
    conn.close()
    print("✅ Job posted.")

# ---------------- QUERIES ----------------
def get_all_jobs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs")
    results = cursor.fetchall()
    conn.close()
    return results

def get_candidates():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM candidate_details")
    results = cursor.fetchall()
    conn.close()
    return results
