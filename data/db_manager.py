import sqlite3
from datetime import datetime
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS frames (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            hash TEXT, confidence REAL, is_fake INTEGER,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS matches (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            source_hash TEXT, target_hash TEXT,
                            similarity REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()

    def add_frame(self, hash_val, confidence, is_fake):
        print(f"[DB DEBUG] Logging frame: hash={hash_val}, confidence={confidence}, is_fake={is_fake}")
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO frames (hash, confidence, is_fake) VALUES (?, ?, ?)",
                      (hash_val, confidence, int(is_fake)))
            conn.commit()
        print(f"[DB DEBUG] Frame logged successfully.")

    def get_matches(self, hash_val):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM matches WHERE source_hash = ? OR target_hash = ?",
                      (hash_val, hash_val))
            return c.fetchall()

    def get_stats(self):
        # Dummy stats for demo purposes
        return {'total_frames': 42, 'fake_frames': 21, 'real_frames': 21}
