"""Database connection and config for MySQL."""
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "room_cleanliness"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": True,
}


def get_connection():
    """Return a new MySQL connection. Caller should close it when done."""
    return pymysql.connect(**MYSQL_CONFIG)
