from pathlib import Path

import mlflow


def start_run(run_name):
    mlflow.set_experiment("mistral-qlora")
    return mlflow.start_run(run_name=run_name)


def log_params(params):
    mlflow.log_params(params)


def log_metrics(trainer):
    for row in trainer.state.log_history:
        step = row.get("step")
        for key in ("loss", "train_loss"):
            if key in row and step is not None:
                mlflow.log_metric(key, row[key], step=step)


def log_artifacts(output_dir):
    output_dir = Path(output_dir)

    if output_dir.exists():
        mlflow.log_artifacts(str(output_dir), artifact_path="outputs")
