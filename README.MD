# Fashion MNIST: Full-Stack ML Pipeline

A production-ready machine learning pipeline that automates training, logs metrics to PostgreSQL, and monitors real-world inference performance using a Streamlit feedback loop.

## 🚀 Features

- **Automated Training Pipeline**: The training script (`models/train.py`) trains a Keras Sequential model and automatically logs architecture, hyperparameters, and accuracy to a SQL database.
- **Experiment Tracking**: Uses **PostgreSQL** to persist training history, allowing for comparison between different optimizers (e.g., SGD vs. Nadam).
- **Inference Monitoring**: A Streamlit interface that logs "Real-World" predictions (e.g., Adidas Samba vs. Bag) to a dedicated `inference_logs` table for error analysis.
- **Infrastructure as Code**: Fully Dockerized database setup with SQLAlchemy 2.0 connection pooling and environment variable configuration.

## 🛠️ Tech Stack

- **ML Core**: TensorFlow/Keras, Scikit-Learn
- **Backend & DB**: Python, PostgreSQL (Dockerized), SQLAlchemy 2.0
- **Frontend**: Streamlit
- **DevOps**: Docker, Dotenv

## 📂 Project Structure

```text
Micro-Project(1)/
├── .env                # Database credentials (not committed)
├── app/
│   └── streamlit_app.py # Inference Interface
├── database/
│   ├── __init__.py     # Package marker
│   ├── db_manager.py   # SQLAlchemy connection logic
│   └── schema.sql      # Table definitions
└── models/
    └── train.py        # Keras training script
```
