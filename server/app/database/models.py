from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class Project(Base):
    __tablename__ = "project"

    id = Column(Integer, primary_key=True)
    name = Column(String(40), nullable=False)
    created_at = Column(DateTime, default=datetime.now())

    sessions = relationship("Session", back_populates="project")


class Session(Base):
    __tablename__ = "session"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now())

    project = relationship("Project", back_populates="sessions")
    prompts = relationship("Prompt", back_populates="session")
    responses = relationship("PromptResponse", back_populates="session")


class Prompt(Base):
    __tablename__ = "prompt"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("session.id"), nullable=False)
    name = Column(String(50), nullable=False)  # e.g. "baseline", "v2"
    system_prompt = Column(Text, nullable=False)

    session = relationship("Session", back_populates="prompts")
    responses = relationship("PromptResponse", back_populates="prompt")


class TestCase(Base):
    __tablename__ = "test_case"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    expected_answer = Column(Text)
    meta = Column(JSON)

    responses = relationship("PromptResponse", back_populates="test_case")


class PromptResponse(Base):
    __tablename__ = "prompt_response"

    id = Column(Integer, primary_key=True)
    test_case_id = Column(Integer, ForeignKey("test_case.id"), nullable=False)
    prompt_id = Column(Integer, ForeignKey("prompt.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("session.id"), nullable=False)

    response = Column(Text, nullable=False)
    model_name = Column(String)
    temperature = Column(Float)
    latency_ms = Column(Integer)
    token_usage = Column(JSON)  # {"prompt": ..., "completion": ...}
    faithfulness = Column(String)
    correctness_score = Column(Float)
    feedback = Column(String)
    created_at = Column(DateTime, default=datetime.now())

    test_case = relationship("TestCase", back_populates="responses")
    prompt = relationship("Prompt", back_populates="responses")
    session = relationship("Session", back_populates="responses")
