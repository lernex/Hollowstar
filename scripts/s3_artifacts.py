from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def join_s3_uri(base_uri: str, *parts: str) -> str:
    bucket, key = parse_s3_uri(base_uri)
    suffix = "/".join(part.strip("/") for part in parts if part)
    joined = f"s3://{bucket}"
    final_key = "/".join(item for item in [key.rstrip("/"), suffix] if item)
    if final_key:
        joined += f"/{final_key}"
    return joined


def iter_local_files(local_dir: Path) -> Iterator[Path]:
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            yield path


class S3ArtifactStore:
    def __init__(self, region_name: str | None = None) -> None:
        session = boto3.session.Session(region_name=region_name)
        self.s3 = session.client("s3")
        multipart_mb = int(os.environ.get("METIS15_S3_MULTIPART_MB", "8"))
        self.transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            multipart_chunksize=max(1, multipart_mb) * 1024 * 1024,
            max_concurrency=max(1, int(os.environ.get("METIS15_S3_MAX_CONCURRENCY", "4"))),
            use_threads=True,
        )

    def ensure_bucket(self, bucket: str, *, region_name: str | None = None) -> None:
        try:
            self.s3.head_bucket(Bucket=bucket)
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
        kwargs = {"Bucket": bucket}
        if region_name and region_name != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region_name}
        self.s3.create_bucket(**kwargs)

    def prefix_exists(self, s3_uri: str) -> bool:
        bucket, prefix = parse_s3_uri(s3_uri)
        response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return bool(response.get("Contents"))

    def upload_file(
        self,
        *,
        local_path: str | Path,
        s3_uri: str,
        content_type: str | None = None,
        extra_metadata: dict[str, str] | None = None,
    ) -> None:
        local_path = Path(local_path)
        bucket, key = parse_s3_uri(s3_uri)
        if content_type is None:
            guessed, _ = mimetypes.guess_type(str(local_path))
            content_type = guessed
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if extra_metadata:
            extra_args["Metadata"] = extra_metadata
        self.s3.upload_file(
            str(local_path),
            bucket,
            key,
            ExtraArgs=extra_args or None,
            Config=self.transfer_config,
        )

    def upload_text(
        self,
        *,
        text: str,
        s3_uri: str,
        content_type: str = "application/json",
    ) -> None:
        bucket, key = parse_s3_uri(s3_uri)
        self.s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)

    def download_file(self, *, s3_uri: str, local_path: str | Path) -> None:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket, key = parse_s3_uri(s3_uri)
        self.s3.download_file(bucket, key, str(local_path), Config=self.transfer_config)

    def upload_dir(self, *, local_dir: str | Path, s3_uri: str) -> int:
        local_dir = Path(local_dir)
        uploaded = 0
        for path in iter_local_files(local_dir):
            rel = path.relative_to(local_dir).as_posix()
            self.upload_file(local_path=path, s3_uri=join_s3_uri(s3_uri, rel))
            uploaded += 1
        return uploaded

    def download_dir(self, *, s3_uri: str, local_dir: str | Path, optional: bool = False) -> int:
        local_dir = Path(local_dir)
        bucket, prefix = parse_s3_uri(s3_uri)
        object_prefix = prefix.rstrip("/") + "/" if prefix else ""
        paginator = self.s3.get_paginator("list_objects_v2")
        downloaded = 0
        saw_any = False
        for page in paginator.paginate(Bucket=bucket, Prefix=object_prefix or prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if object_prefix and not key.startswith(object_prefix):
                    continue
                rel = key[len(object_prefix) :] if object_prefix else key
                if not rel or (rel.endswith("/") and int(item.get("Size", 0)) == 0):
                    continue
                saw_any = True
                target = local_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(bucket, key, str(target), Config=self.transfer_config)
                downloaded += 1
        if not saw_any and not optional:
            raise FileNotFoundError(f"No S3 objects found under {s3_uri}")
        return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Small S3 helper for Metis artifact staging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure_bucket = subparsers.add_parser("ensure-bucket")
    ensure_bucket.add_argument("--s3-uri", required=True)
    ensure_bucket.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))

    upload_file = subparsers.add_parser("upload-file")
    upload_file.add_argument("--local-path", required=True)
    upload_file.add_argument("--s3-uri", required=True)

    upload_dir = subparsers.add_parser("upload-dir")
    upload_dir.add_argument("--local-dir", required=True)
    upload_dir.add_argument("--s3-uri", required=True)

    download_file = subparsers.add_parser("download-file")
    download_file.add_argument("--s3-uri", required=True)
    download_file.add_argument("--local-path", required=True)

    download_dir = subparsers.add_parser("download-dir")
    download_dir.add_argument("--s3-uri", required=True)
    download_dir.add_argument("--local-dir", required=True)
    download_dir.add_argument("--optional", action="store_true")

    write_json = subparsers.add_parser("write-json")
    write_json.add_argument("--json-path", required=True)
    write_json.add_argument("--s3-uri", required=True)

    args = parser.parse_args()
    store = S3ArtifactStore()

    if args.command == "ensure-bucket":
        bucket, _ = parse_s3_uri(args.s3_uri)
        store.ensure_bucket(bucket, region_name=args.region)
        print(bucket, flush=True)
        return

    if args.command == "upload-file":
        store.upload_file(local_path=args.local_path, s3_uri=args.s3_uri)
        print(args.s3_uri, flush=True)
        return

    if args.command == "upload-dir":
        uploaded = store.upload_dir(local_dir=args.local_dir, s3_uri=args.s3_uri)
        print(json.dumps({"s3_uri": args.s3_uri, "uploaded_files": uploaded}), flush=True)
        return

    if args.command == "download-file":
        store.download_file(s3_uri=args.s3_uri, local_path=args.local_path)
        print(args.local_path, flush=True)
        return

    if args.command == "download-dir":
        downloaded = store.download_dir(s3_uri=args.s3_uri, local_dir=args.local_dir, optional=args.optional)
        print(json.dumps({"local_dir": args.local_dir, "downloaded_files": downloaded}), flush=True)
        return

    if args.command == "write-json":
        payload = Path(args.json_path).read_text(encoding="utf-8")
        store.upload_text(text=payload, s3_uri=args.s3_uri, content_type="application/json")
        print(args.s3_uri, flush=True)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
