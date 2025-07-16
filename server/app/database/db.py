from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from app.database.models import Base, Project, Session, Prompt, TestCase, PromptResponse


class Database:
    def __init__(self, url: str = "postgresql+psycopg://postgres:postgres@db:5432"):
        self.engine = create_engine(url, echo=False, future=True)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)

    def create_all(self):
        Base.metadata.create_all(self.engine)
        print("✅ All tables created.")

    def drop_all(self):
        Base.metadata.drop_all(self.engine)
        print("🗑️ All tables dropped.")

    def commit(self):
        self.Session().commit()

    def create_project(self, name: str):
        self.Session.add(Project(name=name))
        self.commit()

    def all_projects(self):
        return self.Session.query(Project).all()

    def get_session(self, project_id):
        return self.Session.query(Session).filter_by(project_id=project_id).all()

    def create_new_session(self, project_id: int):
        print(f"\n\n{project_id}\n\n")
        self.Session.add(
            Session(
                project_id=project_id,
            )
        )
        self.commit()
        return (
            self.Session.query(Session)
            .filter_by(project_id=project_id)
            .first()
            .created_at
        )
