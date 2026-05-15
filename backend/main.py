"""
FastAPI主应用入口
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from backend.config.settings import settings, validate_settings
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    _PROM_AVAILABLE = True
except Exception:
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"  # type: ignore
    _PROM_AVAILABLE = False
from backend.api.kb_router import router as kb_router
from backend.api.agent_router import router as agent_router
from backend.api.chat_router import router as chat_router
from backend.api.stats_router import router as stats_router
from backend.api.config_router import router as config_router
from backend.api.rag_router import router as rag_router
from backend.utils.logger import setup_logger
from backend.dependencies import get_vector_store_adapter, get_knowledge_base_service

# 设置日志
logger = setup_logger(
    "langgraph_rag_backend",
    level=settings.log_level,
    log_file=settings.log_file
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("启动LangGraph多智能体RAG系统后端服务")
    
    # 验证设置
    if not validate_settings():
        logger.error("设置验证失败，请检查配置")
        raise RuntimeError("设置验证失败")
    
    # 初始化服务
    try:
        get_vector_store_adapter()
        get_knowledge_base_service()
        logger.info("服务初始化完成")
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("关闭LangGraph多智能体RAG系统后端服务")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于LangGraph的多智能体RAG工作流系统",
    debug=settings.debug,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# 注册路由
app.include_router(kb_router)
app.include_router(agent_router)
app.include_router(chat_router)
app.include_router(stats_router)
app.include_router(config_router)
app.include_router(rag_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用LangGraph多智能体RAG系统",
        "version": settings.app_version,
        "status": "running"
    }


@app.get("/auth/health")
async def health_check():
    """健康检查"""
    return {
        "status": "success",
        "message": "服务运行正常",
        "version": settings.app_version,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "request_id": ""
        }
    )

# Prometheus metrics endpoint（可选）
if settings.metrics_enabled and _PROM_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        data = generate_latest()
        return JSONResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "内部服务器错误",
            "request_id": ""
        }
    )


if __name__ == "__main__":
    # 运行服务器
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
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        log_config=log_config,
    )
