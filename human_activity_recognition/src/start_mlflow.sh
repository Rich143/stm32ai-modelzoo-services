mlflow ui \
    --backend-store-uri sqlite:///$(pwd)/experiment_outputs/mlruns.db \
    --default-artifact-root $(pwd)/experiment_outputs/artifacts \
    --port 5000 \
    --host 127.0.0.1
