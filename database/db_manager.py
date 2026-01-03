import os
import json
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# 1. Configuration
load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "MYpostsql^123")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5433")  # Port 5433 to bypass local conflict
DB_NAME = os.getenv("DB_NAME", "ml_metadata")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# 2. Database Engine & Session
engine = create_engine(DATABASE_URL, echo=True)
Session = sessionmaker(bind=engine)


def setup_database(filename="database/schema.sql"):
    try:
        with open(filename, "r") as file:
            query = file.read()
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text(query))
            print(f"✅ Table 'training_logs' created/verified in {DB_NAME}.")
    except Exception as e:
        print(f"❌ Setup Error: {e}")


def log_keras_history(model_name, hyperparameters, history):
    # Extract final metrics from history.history dict
    final_metrics = {
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1]),
    }

    insert_query = text(
        """
        INSERT INTO training_logs (model_architecture, hyperparameters, final_loss, final_accuracy)
        VALUES (:arch, :params, :loss, :acc)
    """
    )

    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(
                    insert_query,
                    {
                        "arch": model_name,
                        "params": json.dumps(hyperparameters),
                        "loss": final_metrics["final_loss"],
                        "acc": final_metrics["final_accuracy"],
                    },
                )
            print(f"✅ Successfully logged run for {model_name}")
    except Exception as e:
        print(f"❌ Failed to log metrics: {e}")


if __name__ == "__main__":
    setup_database()
