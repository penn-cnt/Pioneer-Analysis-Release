import os
from sqlalchemy import create_engine
from dotenv import load_dotenv


def get_db_engine():
    """
    Loads database configuration from environment variables
    and returns a SQLAlchemy engine instance.
    """
    load_dotenv()

    DB_USER = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST_LOCAL")
    DB_PORT = os.environ.get("DB_HOST_PORT")
    UNIFIED_DB_NAME = "pioneer-preprint"

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
        raise ValueError(
            "Database environment variables are not set. Please check your .env file."
        )

    # Create the database connection URL
    db_url = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{UNIFIED_DB_NAME}"
    )

    # Create and return a SQLAlchemy engine
    return create_engine(db_url)


if __name__ == "__main__":
    # This block allows you to test the connection directly
    # by running `python db_connector.py` in your terminal.
    try:
        engine = get_db_engine()
        with engine.connect() as connection:
            print("✅ Database connection successful!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
