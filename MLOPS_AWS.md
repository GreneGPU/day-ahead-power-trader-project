# AWS SageMaker + MLflow MLOps

This project now has a deliberately small MLOps path: train the existing transfer-learning model, persist one inference bundle, enforce a promotion gate, track runs in MLflow, and register only passing models in SageMaker Model Registry.

## Local development

Install the optional tooling:

```powershell
python -m pip install -e ".[training,mlops,aws,dev]"
```

Start local MLflow and train:

```powershell
mlflow server --host 127.0.0.1 --port 5000
python -m intraday_power_quant.cli --config configs/jakob-local.json run --enable-mlflow --mlflow-tracking-uri http://127.0.0.1:5000
```

The run creates `power_forecast_bundle.pkl`, `model_gate.json`, forecasts, metrics, and an MLflow run. Pickle artifacts must only be loaded from trusted training runs.

Score prepared features locally:

```powershell
python -m intraday_power_quant.cli score --model-bundle outputs/model_run/power_forecast_bundle.pkl --input prepared_features.csv --output predictions.csv
```

After actual prices arrive, calculate service-quality metrics:

```powershell
python -m intraday_power_quant.cli monitor --forecasts forecasts_with_actuals.csv
```

## SageMaker pipeline

Upload the hourly and 15-minute input files under one S3 prefix, upload the JSON config separately, then create/update the pipeline:

```powershell
python sagemaker_jobs/pipeline.py --role <SAGEMAKER_ROLE_ARN> --region eu-central-1 --data-s3-uri s3://<bucket>/training/ --config-s3-uri s3://<bucket>/config/jakob-local.json --output-s3-uri s3://<bucket>/pipeline-output/ --model-package-group power-trader-models --pipeline-name power-trader-training
```

The pipeline runs:

`S3 data -> SageMaker training -> evaluation of model_gate.json -> conditional Model Registry registration`

Use `sagemaker_jobs/batch_transform.py` for scheduled 15-minute batch forecasts. A real-time endpoint is possible with the same `inference.py`, but batch inference is the cheaper and better fit unless a downstream trader requires low-latency requests.

## What is intentionally not included

Kubeflow, Azure ML, Databricks, and Delta Lake are not added. They would duplicate orchestration or introduce a second cloud. Delta Lake becomes useful once the project has large, frequently updated tables and Spark/Databricks users; it is not needed for the current file-based workload.

## Production checklist

- Put S3 buckets and SageMaker resources in the same AWS region.
- Give the execution role least-privilege access to the exact S3 prefixes, ECR, CloudWatch, SageMaker, and Model Registry group.
- Use AWS-managed MLflow if the team wants SageMaker Studio integration; otherwise a small self-hosted MLflow server is enough initially.
- Add EventBridge scheduling only after the batch command works manually.
- Add cost tags and an AWS Budget before running managed jobs.
- Require a human approval step before deploying a registered model to any automated trading workflow.
