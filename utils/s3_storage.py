"""
Module: utils/s3_storage.py
Purpose: Production S3 & Cloud Storage Manager for downloading project ZIP archives
         and uploading generated CAD drawings and audit reports.
Standards: RDSO Indian Railways BG Standards

Supports:
    1. Direct AWS S3 URIs (s3://bucket-name/key/path/project.zip) via boto3.
    2. Presigned S3 & Cloud HTTPS URLs via streaming requests.
    3. IAM Roles, AWS credentials environment variables, or config overrides.
"""
from __future__ import annotations

import os
import re
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import requests
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception
    BOTO3_AVAILABLE = False


class S3StorageManager:
    """Manages S3 file downloads and uploads for railway project packages."""

    def __init__(
        self,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        download_timeout_s: int = 300,
        s3_url_base: Optional[str] = None,
    ) -> None:
        self.region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "ap-south-1"
        )
        self.aws_access_key_id = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = aws_session_token or os.environ.get("AWS_SESSION_TOKEN")
        self.download_timeout_s = download_timeout_s
        self.s3_url_base = (
            s3_url_base
            or os.environ.get("AWS_S3_URL")
            or ""
        ).rstrip("/")
        self._s3_client = None

    def _get_s3_client(self):
        """Lazy-initialize boto3 S3 client using credentials or AWS default chain."""
        if not BOTO3_AVAILABLE:
            raise RuntimeError(
                "boto3 library is required to interact with direct s3:// URIs. "
                "Install it via: pip install boto3"
            )
        if self._s3_client is None:
            kwargs: dict[str, Any] = {"region_name": self.region_name}
            if self.aws_access_key_id and self.aws_secret_access_key:
                kwargs["aws_access_key_id"] = self.aws_access_key_id
                kwargs["aws_secret_access_key"] = self.aws_secret_access_key
                if self.aws_session_token:
                    kwargs["aws_session_token"] = self.aws_session_token
            self._s3_client = boto3.client("s3", **kwargs)
        return self._s3_client

    @staticmethod
    def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """
        Parse an S3 URI (s3://bucket/key/path/file.zip) into (bucket, key).
        Raises ValueError if format is invalid.
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI format (must start with 's3://'): {s3_uri}")
        parsed = urllib.parse.urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: bucket and key must both be specified in '{s3_uri}'")
        return bucket, key

    def download_file(self, source: str, dest_path: Path) -> Path:
        """
        Download a file from an S3 URI, presigned HTTPS URL, or copy local file.
        
        Args:
            source: 's3://bucket/key.zip', 'https://...', or local filepath
            dest_path: Target Path to save downloaded file
        
        Returns:
            Path to downloaded local file
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        source_clean = source.strip()

        # 1. Direct AWS S3 URI (s3://...)
        if source_clean.startswith("s3://"):
            bucket, key = self.parse_s3_uri(source_clean)
            logger.info("Downloading from S3: bucket='%s', key='%s' -> %s", bucket, key, dest_path)
            s3 = self._get_s3_client()
            try:
                s3.download_file(bucket, key, str(dest_path))
                logger.info("Successfully downloaded S3 object (%d bytes).", dest_path.stat().st_size)
                return dest_path
            except ClientError as e:
                err_code = e.response.get("Error", {}).get("Code", "Unknown")
                err_msg = e.response.get("Error", {}).get("Message", str(e))
                logger.error("S3 ClientError [%s]: %s", err_code, err_msg)
                raise RuntimeError(f"Failed to download from S3 ({source_clean}): [{err_code}] {err_msg}") from e
            except Exception as e:
                logger.error("Unexpected error downloading from S3: %s", str(e), exc_info=True)
                raise RuntimeError(f"S3 download failed for {source_clean}: {str(e)}") from e

        # 2. HTTP / HTTPS URL (Presigned S3 URL or web URL)
        elif source_clean.startswith("http://") or source_clean.startswith("https://"):
            logger.info("Streaming download from HTTP/HTTPS URL -> %s", dest_path)
            try:
                with requests.get(
                    source_clean,
                    stream=True,
                    timeout=(10.0, float(self.download_timeout_s))
                ) as resp:
                    resp.raise_for_status()
                    with dest_path.open("wb") as f_out:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                            if chunk:
                                f_out.write(chunk)
                logger.info("Successfully downloaded URL object (%d bytes).", dest_path.stat().st_size)
                return dest_path
            except requests.RequestException as e:
                logger.error("HTTP download failed: %s", str(e))
                raise RuntimeError(f"Failed to download remote file ({source_clean}): {str(e)}") from e

        # 3. Local filesystem fallback
        else:
            local_p = Path(source_clean).resolve()
            if not local_p.exists():
                raise FileNotFoundError(f"Specified source path not found: {source_clean}")
            if local_p != dest_path.resolve():
                import shutil
                shutil.copy2(local_p, dest_path)
            return dest_path

    def get_public_url(self, bucket: str, key: str) -> str:
        """Format the accessible HTTPS URL for an S3 object."""
        clean_key = key.lstrip("/")
        if self.s3_url_base:
            return f"{self.s3_url_base}/{clean_key}"
        return f"https://{bucket}.s3.{self.region_name}.amazonaws.com/{clean_key}"

    def upload_file(
        self,
        local_path: Path,
        bucket: str,
        key: str,
        content_type: Optional[str] = None,
        as_s3_uri: bool = False,
    ) -> str:
        """
        Upload a single local file to an S3 bucket.
        
        Returns:
            Accessible HTTPS URL (or s3:// URI if as_s3_uri=True)
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local file to upload not found: {local_path}")
        s3 = self._get_s3_client()
        logger.info("Uploading %s to s3://%s/%s...", local_path.name, bucket, key)

        extra_args: dict[str, Any] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        elif local_path.suffix.lower() == ".dxf":
            extra_args["ContentType"] = "application/dxf"
        elif local_path.suffix.lower() == ".pdf":
            extra_args["ContentType"] = "application/pdf"
        elif local_path.suffix.lower() in (".kml", ".xml"):
            extra_args["ContentType"] = "application/vnd.google-earth.kml+xml"

        upload_kwargs: dict[str, Any] = {}
        if extra_args:
            upload_kwargs["ExtraArgs"] = extra_args

        s3.upload_file(str(local_path), bucket, key, **upload_kwargs)

        if as_s3_uri:
            s3_uri = f"s3://{bucket}/{key}"
            logger.info("Upload complete (S3 URI): %s", s3_uri)
            return s3_uri

        public_url = self.get_public_url(bucket, key)
        logger.info("Upload complete (HTTPS URL): %s", public_url)
        return public_url

    def upload_directory(self, local_dir: Path, bucket: str, prefix: str) -> dict[str, str]:
        """
        Recursively upload all files in local_dir to s3://bucket/prefix/
        
        Returns:
            Dict mapping filename to uploaded HTTPS URL
        """
        if not local_dir.exists():
            raise FileNotFoundError(f"Directory to upload not found: {local_dir}")
        uploaded_map: dict[str, str] = {}
        clean_prefix = prefix.strip("/")

        for root, _, files in os.walk(local_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(local_dir).as_posix()
                s3_key = f"{clean_prefix}/{rel_path}" if clean_prefix else rel_path
                url = self.upload_file(file_path, bucket, s3_key)
                uploaded_map[file_path.name] = url

        return uploaded_map
