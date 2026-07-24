"""
DermaSense AI — Logger Utility
Cung cấp logging thống nhất cho toàn bộ pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Tạo logger với format chuẩn, output ra cả console và file.

    Args:
        name: Tên logger (thường là __name__ của module gọi).
        log_dir: Đường dẫn thư mục lưu log file. None = chỉ console.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Tránh tạo handler trùng nếu logger đã được configure
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format chung
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (nếu có log_dir)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{name}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
