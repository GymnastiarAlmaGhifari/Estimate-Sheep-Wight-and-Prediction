from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

def use_engine():
    db_uri = f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
    engine = create_engine(db_uri)
    return engine