"""
Sentinel AI - File Handler Utility
Handles file uploads, validation, and cleanup.
"""
import os
import time
import uuid
import aiofiles
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException

from app.config import settings


# Allowed file extensions by type
ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png"},
    "audio": {".mp3", ".wav", ".m4a", ".ogg"},
    "video": {".mp4", ".mov", ".avi", ".webm"}
}

# MIME type mappings
MIME_TYPES = {
    "image": {"image/jpeg", "image/png"},
    "audio": {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/ogg"},
    "video": {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
}


def validate_file_type(file: UploadFile, expected_type: str) -> bool:
    """
    Validate that a file matches the expected type.
    
    Args:
        file: The uploaded file
        expected_type: One of 'image', 'audio', 'video'
        
    Returns:
        True if valid, raises HTTPException otherwise
    """
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS.get(expected_type, set()):
        allowed = ", ".join(ALLOWED_EXTENSIONS[expected_type])
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed extensions for {expected_type}: {allowed}"
        )
    
    return True


async def save_upload(file: UploadFile, file_type: str) -> Tuple[Path, str]:
    """
    Save an uploaded file to disk.
    
    Args:
        file: The uploaded file
        file_type: Type of file (image, audio, video)
        
    Returns:
        Tuple of (file_path, file_id)
    """
    # Validate file type
    validate_file_type(file, file_type)
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = f"{file_id}{ext}"
    file_path = settings.upload_dir / filename
    
    # Check file size while saving
    total_size = 0
    max_size = settings.max_upload_size_mb * 1024 * 1024  # Convert to bytes
    
    async with aiofiles.open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            total_size += len(chunk)
            if total_size > max_size:
                # Clean up partial file
                await f.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
                )
            await f.write(chunk)
    
    return file_path, file_id


def delete_file(file_path: Path) -> bool:
    """
    Delete a file from disk.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if deleted, False if not found
    """
    try:
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False


def cleanup_old_files() -> int:
    """
    Clean up files older than the retention period.
    
    Returns:
        Number of files deleted
    """
    deleted = 0
    current_time = time.time()
    retention = settings.file_retention_seconds
    
    if not settings.upload_dir.exists():
        return 0
    
    for file_path in settings.upload_dir.iterdir():
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > retention:
                if delete_file(file_path):
                    deleted += 1
    
    if deleted > 0:
        print(f"🧹 Cleaned up {deleted} old files")
    
    return deleted


def get_file_info(file_path: Path) -> Optional[dict]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict with file info or None if not found
    """
    if not file_path.exists():
        return None
    
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": stat.st_ctime,
        "modified": stat.st_mtime
    }
