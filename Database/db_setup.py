# db_setup.py
import sqlite3

def init_db():
    conn = sqlite3.connect("recruitment.db")
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT CHECK(role IN ('recruiter', 'candidate')) NOT NULL
    )
    """)

    # Candidate details table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidate_details (
        candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        full_name TEXT,
        candidate_title TEXT,
        about TEXT,
        location TEXT,
        preferred_employment_type TEXT,
        yoe REAL,
        seniority TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )
    """)


    # Jobs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id INTEGER PRIMARY KEY AUTOINCREMENT,
        posted_by INTEGER NOT NULL,
        topic TEXT,
        job_title TEXT,
        site TEXT,
        tasks TEXT,
        perks_benefits TEXT,
        skills_tech_stack TEXT,
        educational_requirements TEXT,
        seniority TEXT,
        yoe REAL,
        employment_type TEXT,
        extra_info TEXT,
        created_at TEXT,
        FOREIGN KEY(posted_by) REFERENCES users(user_id)
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("✅ Database initialized successfully.")
