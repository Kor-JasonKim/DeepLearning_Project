"""Create database and tables. Run once: python init_db.py"""
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

# Connect without database to create it if needed
host = os.environ.get("MYSQL_HOST", "localhost")
user = os.environ.get("MYSQL_USER", "root")
password = os.environ.get("MYSQL_PASSWORD", "")
database = os.environ.get("MYSQL_DATABASE", "room_cleanliness")

def main():
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        conn.select_db(database)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    display_name VARCHAR(255) NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NULL,
                    image_path VARCHAR(512) NOT NULL,
                    ai_score FLOAT NOT NULL,
                    mode VARCHAR(32) NOT NULL DEFAULT 'room',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_scores (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    post_id INT NOT NULL,
                    score FLOAT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_user_post (user_id, post_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
                )
            """)
        conn.commit()
        print(f"Database '{database}' and tables ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
