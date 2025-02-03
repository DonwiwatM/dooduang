import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
database = os.getenv("DATABASE")
port = os.getenv("PORT")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")

SQLALCHEMY_DATABASE_URL = f"postgresql://{database}:{password}@{host}:{port}/{database}"
print(SQLALCHEMY_DATABASE_URL)

# Create engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={})

Base = declarative_base()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

class UserCreate(BaseModel):
    name: str
    email: str

Base.metadata.create_all(bind=engine)
