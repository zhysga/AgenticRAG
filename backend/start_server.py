#!/usr/bin/env python3
"""
后端服务启动脚本
"""
import uvicorn
import os
import sys
from pathlib import Path

# 添加backend目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from backend.config.settings import settings
    # ensure log dir exists
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": False,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": "%(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": settings.log_file,
                "encoding": "utf-8",
            },
            "access_file": {
                "class": "logging.FileHandler",
                "formatter": "access",
                "filename": settings.log_file,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["file"], "level": settings.log_level.upper(), "propagate": False},
            "uvicorn.error": {"handlers": ["file"], "level": settings.log_level.upper(), "propagate": False},
            "uvicorn.access": {"handlers": ["access_file"], "level": settings.log_level.upper(), "propagate": False},
        },
    }

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        log_config=log_config,
    )
