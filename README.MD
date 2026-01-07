# 👗 Fashion MNIST: Full-Stack ML Pipeline

> A production-ready machine learning pipeline with automated training, PostgreSQL experiment tracking, and real-time inference monitoring.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-336791.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)

---

## 🎯 Overview

This project demonstrates a complete MLOps workflow for Fashion MNIST classification, featuring automated experiment tracking, database-backed model registry, and a production-grade inference pipeline with feedback collection.

### Key Highlights

- **🤖 Automated Training**: Keras Sequential model with configurable optimizers
- **📊 Experiment Tracking**: PostgreSQL-backed metrics storage for model comparison
- **🔍 Inference Monitoring**: Real-time prediction logging and error analysis
- **🐳 Production-Ready**: Fully containerized database with SQLAlchemy 2.0
- **⚡ Feedback Loop**: Streamlit UI for capturing real-world performance metrics

---

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Training Script│─────▶│  PostgreSQL DB   │◀─────│ Streamlit App   │
│   (train.py)    │      │  - training_logs │      │ (inference UI)  │
│                 │      │  - inference_logs│      │                 │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

---

## 🛠️ Tech Stack

| Category         | Technologies                   |
| ---------------- | ------------------------------ |
| **ML Framework** | TensorFlow/Keras, Scikit-Learn |
| **Database**     | PostgreSQL 13+ (Docker)        |
| **ORM**          | SQLAlchemy 2.0                 |
| **Frontend**     | Streamlit                      |
| **DevOps**       | Docker, python-dotenv          |

---

## 📁 Project Structure

```
Micro-Project(1)/
│
├── 📄 .env                    # Environment configuration (gitignored)
├── 📄 .gitignore              # Files to exclude from Git
├── 📄 docker-compose.yml      # Container orchestration for DB
├── 📄 requirements.txt        # Python dependencies
│
├── 📂 app/
│   └── streamlit_app.py       # Inference interface with feedback collection
│
├── 📂 database/
│   ├── __init__.py            # Package initialization
│   ├── db_manager.py          # SQLAlchemy connection & session management
│   └── schema.sql             # Database schema definitions
│
├── 📂 models/
│   ├── custom_loss.py         # Extra code (optional)
│   └── train.py               # Model training & experiment logging
│
└── 📂 notebooks/
    └── exploration.ipynb  # Jupyter notebook for prototyping
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker
- pip

### 1️⃣ Clone & Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd Micro-Project\(1\)

# Install Python packages
pip install -r requirements.txt
```

### 2️⃣ Launch PostgreSQL Database

```bash
# Start PostgreSQL container (port 5433 to avoid conflicts)
docker run --name ml_db \
  -e POSTGRES_PASSWORD=<your_password> \
  -d -p 5433:5432 \
  postgres

# Create database
docker exec -it ml_db psql -U <your_username> -c "CREATE DATABASE <your_database_name>;"
```

### 3️⃣ Configure Environment

Create a `.env` file in the project root:

```env
DB_USER=<your_username>
DB_PASSWORD=<your_password>
DB_HOST=<your_host>
DB_PORT=<your_port>
DB_NAME=<your_database_name>
```

### 4️⃣ Initialize Database Schema

```bash
python database/db_manager.py
```

---

## 💻 Usage

### Training the Model

Run the training pipeline to train the Fashion MNIST classifier:

```bash
python -m models.train
```

**What happens:**

- Loads Fashion MNIST dataset
- Trains Keras Sequential model
- Logs hyperparameters, architecture, and metrics to `training_logs`
- Saves model artifacts

### Running Inference Interface

Launch the Streamlit app for interactive predictions:

```bash
streamlit run app/streamlit_app.py
```

**Features:**

- Upload fashion item images
- Get real-time predictions with confidence scores
- Log predictions to `inference_logs` for monitoring
- Analyze model performance on edge cases

---

## 📊 Database Schema

### `training_logs`

| Column            | Type        | Description                            |
| ----------------- | ----------- | -------------------------------------- |
| `run_id`          | VARCHAR(50) | Unique training run identifier         |
| `timestamp`       | TIMESTAMP   | Training completion time               |
| `hyperparameters` | JSON        | Optimizer, learning rate, epochs, etc. |
| `final_accuracy`  | FLOAT       | Test set accuracy                      |

### `inference_logs`

| Column             | Type        | Description                   |
| ------------------ | ----------- | ----------------------------- |
| `id`               | SERIAL      | Auto-incrementing primary key |
| `predicted_class`  | VARCHAR(50) | Model prediction label        |
| `confidence_score` | FLOAT       | Prediction confidence (0-1)   |
| `timestamp`        | TIMESTAMP   | Inference time                |

---

## 🔧 Configuration Options

### Model Hyperparameters

Edit `models/train.py` to customize:

```python
EPOCHS = 10
BATCH_SIZE = 128
OPTIMIZER = 'nadam'  # Options: 'sgd', 'adam', 'nadam'
LEARNING_RATE = 0.001
```

### Database Connection

Connection pooling is configured in `database/db_manager.py`:

```python
pool_size=5
max_overflow=10
pool_timeout=30
```

---

## 📈 Example Workflow

```bash
# 1. Train multiple models with different optimizers
python -m models.train  # Run with SGD
# Edit train.py to use 'nadam'
python -m models.train  # Run with Nadam

# 2. Compare results in PostgreSQL
docker exec -it ml_db psql -U postgres -d ml_metadata
SELECT run_id, hyperparameters->>'optimizer' AS optimizer,
       final_accuracy
FROM training_logs
ORDER BY final_accuracy DESC;

# 3. Test inference and monitor predictions
streamlit run app/streamlit_app.py
```

---

## 🐛 Troubleshooting

### Port Conflict

If port 5433 is already in use:

```bash
# Use a different port
docker run --name ml_db -e POSTGRES_PASSWORD=<your_password> -d -p 5434:5432 postgres

# Update .env accordingly
DB_PORT=<your_port>
```

### Connection Errors

```bash
# Check if container is running
docker ps

# View container logs
docker logs ml_db

# Restart container
docker restart ml_db
```

---

## 📝 Future Enhancements

- [ ] Add model versioning with MLflow
- [ ] Implement A/B testing framework
- [ ] Add Prometheus metrics export
- [ ] Create REST API with FastAPI
- [ ] Add automated retraining pipeline
- [ ] Implement data drift detection

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

<p align="center">Made with ❤️ for MLOps learners</p>
