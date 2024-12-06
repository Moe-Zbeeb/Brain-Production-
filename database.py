# database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from base import Base

# Import all models so Base is aware of them before creating tables
from models import User, Course, CourseFile, StudentQuestion

# Database connection URL
DATABASE_URL = "sqlite:///./app_database.db"  # Replace with your desired path

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite-specific
)

# Create a configured session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables in the database defined by Base's subclasses
Base.metadata.create_all(bind=engine)
