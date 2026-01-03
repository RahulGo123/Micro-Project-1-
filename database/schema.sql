CREATE TABLE IF NOT EXISTS training_logs(
    run_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_architecture TEXT,
    hyperparameters JSONB,
    final_loss FLOAT,
    final_accuracy FLOAT
)