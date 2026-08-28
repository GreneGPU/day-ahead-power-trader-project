from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "sagemaker_jobs"
PACKAGE_DEPENDENCY = ROOT / "src"


def build_pipeline(
    *,
    role: str,
    region: str,
    data_s3_uri: str,
    config_s3_uri: str,
    output_s3_uri: str,
    model_package_group_name: str,
    pipeline_name: str = "power-trader-training",
    framework_version: str = "1.4-2",
    training_instance_type: str = "ml.m5.xlarge",
    processing_instance_type: str = "ml.m5.large",
    mlflow_tracking_uri: str | None = None,
    mlflow_registered_model_name: str | None = None,
):
    try:
        import boto3
        from sagemaker.inputs import TrainingInput
        from sagemaker.processing import ProcessingInput, ProcessingOutput
        from sagemaker.sklearn.estimator import SKLearn
        from sagemaker.sklearn.model import SKLearnModel
        from sagemaker.sklearn.processing import SKLearnProcessor
        from sagemaker.workflow.condition_step import ConditionStep
        from sagemaker.workflow.conditions import ConditionEquals
        from sagemaker.workflow.functions import JsonGet
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.pipeline_context import PipelineSession
        from sagemaker.workflow.properties import PropertyFile
        from sagemaker.workflow.steps import ModelStep, ProcessingStep, TrainingStep
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Install the project with the "aws" extra to build the SageMaker pipeline.'
        ) from exc

    session = PipelineSession(boto_session=boto3.Session(region_name=region))
    hyperparameters = {
        "enable-mlflow": str(bool(mlflow_tracking_uri)).lower(),
        "mlflow-experiment": "power-trader-dk1-15min",
    }
    if mlflow_tracking_uri:
        hyperparameters["mlflow-tracking-uri"] = mlflow_tracking_uri
    if mlflow_registered_model_name:
        hyperparameters["mlflow-register-model"] = mlflow_registered_model_name

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(SOURCE_DIR),
        dependencies=[str(PACKAGE_DEPENDENCY)],
        role=role,
        framework_version=framework_version,
        py_version="py3",
        instance_type=training_instance_type,
        instance_count=1,
        output_path=output_s3_uri,
        hyperparameters=hyperparameters,
        sagemaker_session=session,
    )
    train_step = TrainingStep(
        name="TrainPowerForecast",
        step_args=estimator.fit(
            inputs={
                "train": TrainingInput(data_s3_uri),
                "config": TrainingInput(config_s3_uri),
            }
        ),
    )

    processor = SKLearnProcessor(
        framework_version=framework_version,
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=session,
    )
    evaluation_file = PropertyFile(
        name="PowerForecastEvaluation",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluate_step = ProcessingStep(
        name="EvaluatePowerForecast",
        step_args=processor.run(
            code=str(SOURCE_DIR / "evaluate.py"),
            inputs=[
                ProcessingInput(
                    source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=f"{output_s3_uri.rstrip('/')}/evaluation",
                )
            ],
            property_files=[evaluation_file],
        ),
    )

    inference_model = SKLearnModel(
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        source_dir=str(SOURCE_DIR),
        dependencies=[str(PACKAGE_DEPENDENCY)],
        framework_version=framework_version,
        py_version="py3",
        sagemaker_session=session,
    )
    register_step = ModelStep(
        name="RegisterPowerForecast",
        step_args=inference_model.register(
            content_types=["application/json", "text/csv"],
            response_types=["application/json", "text/csv"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status="PendingManualApproval",
        ),
    )
    condition_step = ConditionStep(
        name="PromoteOnlyIfGatePasses",
        conditions=[
            ConditionEquals(
                left=JsonGet(
                    step_name=evaluate_step.name,
                    property_file=evaluation_file,
                    json_path="gate.passed",
                ),
                right=True,
            )
        ],
        if_steps=[register_step],
        else_steps=[],
    )
    return Pipeline(
        name=pipeline_name,
        parameters=[],
        steps=[train_step, evaluate_step, condition_step],
        sagemaker_session=session,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update the SageMaker training pipeline.")
    parser.add_argument("--role", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-s3-uri", required=True)
    parser.add_argument("--config-s3-uri", required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--model-package-group", required=True)
    parser.add_argument("--pipeline-name", default="power-trader-training")
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-register-model")
    args = parser.parse_args()
    pipeline = build_pipeline(
        role=args.role,
        region=args.region,
        data_s3_uri=args.data_s3_uri,
        config_s3_uri=args.config_s3_uri,
        output_s3_uri=args.output_s3_uri,
        model_package_group_name=args.model_package_group,
        pipeline_name=args.pipeline_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_registered_model_name=args.mlflow_register_model,
    )
    pipeline.upsert(role_arn=args.role)
    print("SageMaker pipeline created:", args.pipeline_name)


if __name__ == "__main__":
    main()
