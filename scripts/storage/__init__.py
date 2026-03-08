from .base import Storage
from .s3 import S3Storage
from .local import LocalStorage

__all__ = ["Storage", "S3Storage", "LocalStorage"]
