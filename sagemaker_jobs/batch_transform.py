from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "sagemaker_jobs"
PACKAGE_DEPENDENCY = ROOT / "src"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scheduled SageMaker batch forecasting.")
    parser.add_argument("--role", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model-data", required=True, help="S3 URI for model.tar.gz")
    parser.add_argument("--input-s3-uri", required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--framework-version", default="1.4-2")
    parser.add_argument("--content-type", default="application/json")
    parser.add_argument("--accept", default="application/json")
    args = parser.parse_args()

    try:
        import boto3
        from sagemaker.session import Session
        from sagemaker.sklearn.model import SKLearnModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Install the project with the "aws" extra to start batch inference.'
        ) from exc

    session = Session(boto_session=boto3.Session(region_name=args.region))
    model = SKLearnModel(
        model_data=args.model_data,
        role=args.role,
        entry_point="inference.py",
        source_dir=str(SOURCE_DIR),
        dependencies=[str(PACKAGE_DEPENDENCY)],
        framework_version=args.framework_version,
        py_version="py3",
        sagemaker_session=session,
    )
    transformer = model.transformer(
        instance_count=1,
        instance_type=args.instance_type,
        output_path=args.output_s3_uri,
        accept=args.accept,
        assemble_with="Line",
    )
    transformer.transform(
        data=args.input_s3_uri,
        content_type=args.content_type,
        split_type="Line" if "json" in args.content_type else None,
    )
    transformer.wait()
    print("Batch predictions:", args.output_s3_uri)


if __name__ == "__main__":
    main()
