"""
Storage service for filesystem abstraction.

This module provides a clean interface for file operations,
abstracting away the complexity of filesystem management.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Union, Optional, List
from datetime import datetime

from ..config import Config

logger = logging.getLogger(__name__)


class StorageService:
    """
    Filesystem abstraction service for wound segmentation.
    
    Provides a clean interface for file operations with automatic
    directory management and organized storage structure.
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the storage service.
        
        Args:
            base_dir: Base directory for storage (defaults to Config.OUTPUT_DIR)
        """
        self.base_dir = Path(base_dir) if base_dir else Path(Config.OUTPUT_DIR)
        self._ensure_base_dir()
        
        # Define storage structure
        self.dirs = {
            'reports': self.base_dir / 'reports',
            'uploads': self.base_dir / 'uploads',
            'masks': self.base_dir / 'masks',
            'visualizations': self.base_dir / 'visualizations',
            'voice': self.base_dir / 'voice_summaries',
            'temp': self.base_dir / 'temp'
        }
        
        # Create all directories
        self._create_directories()
        
        logger.info(f"StorageService initialized with base directory: {self.base_dir}")
    
    def _ensure_base_dir(self):
        """Ensure the base directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_directories(self):
        """Create all required subdirectories."""
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")
    
    def save_image(self, image_data: bytes, subdir: str, filename: str) -> Path:
        """
        Save image data to storage.
        
        Args:
            image_data: Image data as bytes
            subdir: Subdirectory name (reports, masks, visualizations, etc.)
            filename: Filename for the image
            
        Returns:
            Path: Full path to saved file
            
        Raises:
            ValueError: If subdir is not valid
            OSError: If file cannot be saved
        """
        if subdir not in self.dirs:
            raise ValueError(f"Invalid subdirectory: {subdir}. Valid options: {list(self.dirs.keys())}")
        
        file_path = self.dirs[subdir] / filename
        
        try:
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Image saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save image {filename}: {e}")
            raise OSError(f"Cannot save image: {e}")
    
    def save_pdf(self, pdf_data: bytes, subdir: str, filename: str) -> Path:
        """
        Save PDF data to storage.
        
        Args:
            pdf_data: PDF data as bytes
            subdir: Subdirectory name
            filename: Filename for the PDF
            
        Returns:
            Path: Full path to saved file
        """
        if subdir not in self.dirs:
            raise ValueError(f"Invalid subdirectory: {subdir}. Valid options: {list(self.dirs.keys())}")
        
        file_path = self.dirs[subdir] / filename
        
        try:
            with open(file_path, 'wb') as f:
                f.write(pdf_data)
            
            logger.info(f"PDF saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save PDF {filename}: {e}")
            raise OSError(f"Cannot save PDF: {e}")
    
    def save_audio(self, audio_data: bytes, filename: str) -> Path:
        """
        Save audio data to voice summaries directory.
        
        Args:
            audio_data: Audio data as bytes
            filename: Filename for the audio file
            
        Returns:
            Path: Full path to saved file
        """
        file_path = self.dirs['voice'] / filename
        
        try:
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"Audio saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save audio {filename}: {e}")
            raise OSError(f"Cannot save audio: {e}")
    
    def save_text(self, text_data: str, subdir: str, filename: str) -> Path:
        """
        Save text data to storage.
        
        Args:
            text_data: Text content
            subdir: Subdirectory name
            filename: Filename for the text file
            
        Returns:
            Path: Full path to saved file
        """
        if subdir not in self.dirs:
            raise ValueError(f"Invalid subdirectory: {subdir}. Valid options: {list(self.dirs.keys())}")
        
        file_path = self.dirs[subdir] / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_data)
            
            logger.info(f"Text saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save text {filename}: {e}")
            raise OSError(f"Cannot save text: {e}")
    
    def load_file(self, subdir: str, filename: str) -> bytes:
        """
        Load file data from storage.
        
        Args:
            subdir: Subdirectory name
            filename: Filename to load
            
        Returns:
            bytes: File data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            OSError: If file cannot be read
        """
        if subdir not in self.dirs:
            raise ValueError(f"Invalid subdirectory: {subdir}. Valid options: {list(self.dirs.keys())}")
        
        file_path = self.dirs[subdir] / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            logger.debug(f"File loaded: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load file {filename}: {e}")
            raise OSError(f"Cannot load file: {e}")
    
    def list_files(self, subdir: str, pattern: str = "*") -> List[Path]:
        """
        List files in a subdirectory.
        
        Args:
            subdir: Subdirectory name
            pattern: File pattern to match (default: all files)
            
        Returns:
            List of file paths
        """
        if subdir not in self.dirs:
            raise ValueError(f"Invalid subdirectory: {subdir}. Valid options: {list(self.dirs.keys())}")
        
        directory = self.dirs[subdir]
        if not directory.exists():
            return []
        
        files = list(directory.glob(pattern))
        logger.debug(f"Found {len(files)} files in {subdir} matching {pattern}")
        return files
    
    def delete_file(self, subdir: str, filename: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            subdir: Subdirectory name
            filename: Filename to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        if subdir not in self.dirs:
            raise ValueError(f"Invalid subdirectory: {subdir}. Valid options: {list(self.dirs.keys())}")
        
        file_path = self.dirs[subdir] / filename
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file {filename}: {e}")
            return False
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age of temp files in hours
            
        Returns:
            int: Number of files cleaned up
        """
        temp_dir = self.dirs['temp']
        if not temp_dir.exists():
            return 0
        
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        try:
            for file_path in temp_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned up temp file: {file_path}")
            
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return 0
    
    def get_storage_info(self) -> dict:
        """
        Get information about storage usage.
        
        Returns:
            dict: Storage information including directory sizes
        """
        info = {
            'base_directory': str(self.base_dir),
            'directories': {},
            'total_size_bytes': 0
        }
        
        for name, path in self.dirs.items():
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                file_count = len(list(path.rglob('*')))
                info['directories'][name] = {
                    'path': str(path),
                    'size_bytes': size,
                    'file_count': file_count
                }
                info['total_size_bytes'] += size
            else:
                info['directories'][name] = {
                    'path': str(path),
                    'size_bytes': 0,
                    'file_count': 0
                }
        
        return info
    
    def create_session_dir(self, session_id: str) -> Path:
        """
        Create a session-specific directory for organizing related files.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Path: Path to the session directory
        """
        session_dir = self.base_dir / 'sessions' / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created session directory: {session_dir}")
        return session_dir


# Global storage service instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """
    Get the global storage service instance.
    
    Returns:
        StorageService: The global storage service instance
    """
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service


def reset_storage_service() -> None:
    """Reset the global storage service (useful for testing)."""
    global _storage_service
    _storage_service = None
    logger.info("Storage service reset")