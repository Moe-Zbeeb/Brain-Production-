# models.py

from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime
from sqlalchemy.orm import relationship
from base import Base
from datetime import datetime
import bcrypt

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'professor' or 'student'
    
    # Relationships
    courses = relationship("Course", back_populates="professor", cascade="all, delete-orphan")
    questions = relationship("StudentQuestion", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class Course(Base):
    __tablename__ = 'courses'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    professor_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    youtube_link = Column(String, nullable=True)  # New column for YouTube links
    
    # Relationships
    professor = relationship("User", back_populates="courses")
    files = relationship("CourseFile", back_populates="course", cascade="all, delete-orphan")
    questions = relationship("StudentQuestion", back_populates="course", cascade="all, delete-orphan")

class CourseFile(Base):
    __tablename__ = 'course_files'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    data = Column(LargeBinary, nullable=False)
    course_id = Column(Integer, ForeignKey('courses.id'), nullable=False)
    course = relationship("Course", back_populates="files")

class StudentQuestion(Base):
    __tablename__ = 'student_questions'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    course_id = Column(Integer, ForeignKey('courses.id'), nullable=False)
    question = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="questions")
    course = relationship("Course", back_populates="questions")