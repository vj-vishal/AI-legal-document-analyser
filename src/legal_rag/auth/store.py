# auth/store.py
import uuid

users_db: dict = {}
# Structure: { email: { "user_id": "uuid", "password_hash": "..." } }

def create_user(email: str, password_hash: str) -> dict:
    user_id = str(uuid.uuid4())
    users_db[email] = {"user_id": user_id, "password_hash": password_hash}
    return users_db[email]

def get_user(email: str) -> dict | None:
    return users_db.get(email)