from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from pydantic import BaseModel
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class UserLog(Base):
    __tablename__ = "user_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)
    details = Column(Text)

class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_id = Column(String, index=True)
    is_preferred = Column(Boolean)
    comments = Column(Text, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models for request/response
class UserLogCreate(BaseModel):
    user_id: str
    action: str
    details: str

class FeedbackCreate(BaseModel):
    user_id: str
    response_id: str
    is_preferred: bool
    comments: str = None

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

@app.post("/log")
async def create_user_log(log: UserLogCreate, db: Session = Depends(get_db)):
    db_log = UserLog(**log.dict())
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return {"status": "success", "log_id": db_log.id}

@app.post("/feedback")
async def create_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    db_feedback = Feedback(**feedback.dict())
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return {"status": "success", "feedback_id": db_feedback.id}

@app.get("/logs/{user_id}")
async def get_user_logs(user_id: str, db: Session = Depends(get_db)):
    logs = db.query(UserLog).filter(UserLog.user_id == user_id).all()
    return logs

@app.get("/feedback/{user_id}")
async def get_user_feedback(user_id: str, db: Session = Depends(get_db)):
    feedback = db.query(Feedback).filter(Feedback.user_id == user_id).all()
    return feedback

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)