import os
from functools import lru_cache
from typing import BinaryIO

import boto3
from botocore.exceptions import BotoCoreError, ClientError


class ObjectStorageError(Exception):
    """Raised when an S3-compatible object-storage operation cannot complete."""


class ObjectStorage:
    def __init__(self) -> None:
        self.bucket = required_environment_variable("S3_BUCKET")
        self.client = boto3.client(
            "s3",
            endpoint_url=required_environment_variable("S3_ENDPOINT_URL"),
            aws_access_key_id=required_environment_variable("S3_ACCESS_KEY_ID"),
            aws_secret_access_key=required_environment_variable("S3_SECRET_ACCESS_KEY"),
            region_name=os.getenv("S3_REGION", "us-east-1"),
        )

    def upload_file(self, file: BinaryIO, object_key: str, content_type: str) -> None:
        try:
            self._ensure_bucket()
            self.client.upload_fileobj(
                file,
                self.bucket,
                object_key,
                ExtraArgs={"ContentType": content_type},
            )
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError("Object storage upload failed.") from error

    def delete_file(self, object_key: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=object_key)
        except (BotoCoreError, ClientError):
            # The database record is the source of truth. A failed compensation delete is not a
            # successful upload and will be handled manually until document cleanup is introduced.
            pass

    def download_file(self, object_key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError("Object storage download failed.") from error

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code")
            if error_code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except ClientError as create_error:
                if create_error.response.get("Error", {}).get("Code") not in {
                    "BucketAlreadyExists",
                    "BucketAlreadyOwnedByYou",
                }:
                    raise


def required_environment_variable(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ObjectStorageError(f"Missing required storage configuration: {name}.")
    return value


@lru_cache
def get_storage_client() -> ObjectStorage:
    return ObjectStorage()
