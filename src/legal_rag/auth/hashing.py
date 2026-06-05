from passlib.context import CryptContext

# Tell passlib to use bcrypt as the hashing algorithm
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """
    Takes a plain text password like "mypassword123"
    Returns a bcrypt hash like "$2b$12$eImiTXuWVxfM37uY4JANjQ..."
    """
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Takes the plain password the user just typed
    and the hash stored in your DB.
    Returns True if they match, False if not.
    Called during login ONLY.
    """
    return pwd_context.verify(plain_password, hashed_password)

if __name__ == "__main__":
    # Example usage
    password = "mypd123"
    hashed = hash_password(password)
    print(f"Hashed password: {hashed}")

    # Verify the password
    is_valid = verify_password("mypd123", hashed)
    print(f"Password valid: {is_valid}")

    # Verify with a wrong password
    is_valid_wrong = verify_password("wpword", hashed)
    print(f"Wrong password valid: {is_valid_wrong}")
    