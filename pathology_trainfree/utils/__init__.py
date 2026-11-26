"""Feature caching and evaluation metrics."""

import hashlib


def sha256(path):
    """Compute the SHA-256 checksum of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
