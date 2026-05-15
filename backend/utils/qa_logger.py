"""
QA协作日志记录工具
记录问答协作过程中的所有智能体交互、中间步骤、引用等详细信息
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class QACollaborationLogger:
    """QA协作日志记录器"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "qa_collaboration.jsonl"
        self._lock = Lock()
        
    def log_qa_session(
        self,
        query_id: str,
        session_id: str,
        query: str,
        workflow_stages: List[Dict[str, Any]],
        intermediate_steps: List[Dict[str, Any]],
        final_answer: str,
        citations: List[Dict[str, Any]],
        total_duration_ms: float,
        status: str = "success",
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录完整的QA会话"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query_id": query_id,
                "session_id": session_id,
                "query": query,
                "workflow_stages": workflow_stages,
                "intermediate_steps": intermediate_steps,
                "final_answer": final_answer,
                "citations": citations,
                "total_duration_ms": total_duration_ms,
                "status": status,
                "error_message": error_message,
                "metadata": metadata or {}
            }
            
            self._write_log_entry(log_entry)
            
        except Exception as e:
            logger.error(f"记录QA会话日志失败: {e}")
    
    def log_agent_execution(
        self,
        query_id: str,
        agent_name: str,
        agent_type: str,
        input_content: str,
        output_content: str,
        reasoning: str,
        confidence: float,
        execution_time: float,
        tools_used: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录单个智能体执行详情"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "log_type": "agent_execution",
                "query_id": query_id,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "input_content": input_content[:500] + "..." if len(input_content) > 500 else input_content,
                "output_content": output_content[:1000] + "..." if len(output_content) > 1000 else output_content,
                "reasoning": reasoning,
                "confidence": confidence,
                "execution_time_ms": execution_time * 1000,
                "tools_used": tools_used,
                "metadata": metadata or {}
            }
            
            self._write_log_entry(log_entry)
            
        except Exception as e:
            logger.error(f"记录智能体执行日志失败: {e}")
    
    def log_workflow_stage(
        self,
        query_id: str,
        stage_name: str,
        stage_data: Dict[str, Any],
        duration_ms: float
    ) -> None:
        """记录工作流阶段"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "log_type": "workflow_stage",
                "query_id": query_id,
                "stage_name": stage_name,
                "stage_data": stage_data,
                "duration_ms": duration_ms
            }
            
            self._write_log_entry(log_entry)
            
        except Exception as e:
            logger.error(f"记录工作流阶段日志失败: {e}")
    
    def _write_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """写入日志条目（线程安全）"""
        with self._lock:
            try:
                # 处理datetime对象的序列化
                def _json_default(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return str(obj)
                
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    json.dump(log_entry, f, ensure_ascii=False, default=_json_default)
                    f.write('\n')
            except Exception as e:
                logger.error(f"写入日志文件失败: {e}")
    
    def get_session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定会话的所有日志"""
        logs = []
        try:
            if not self.log_file.exists():
                return logs
                
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if log_entry.get("session_id") == session_id:
                            logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"读取会话日志失败: {e}")
        
        return logs
    
    def get_query_logs(self, query_id: str) -> List[Dict[str, Any]]:
        """获取指定查询的所有日志"""
        logs = []
        try:
            if not self.log_file.exists():
                return logs
                
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if log_entry.get("query_id") == query_id:
                            logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"读取查询日志失败: {e}")
        
        return logs


# 全局单例
_qa_logger: Optional[QACollaborationLogger] = None


def get_qa_logger(log_dir: str = "./logs") -> QACollaborationLogger:
    """获取QA协作日志记录器单例"""
    global _qa_logger
    if _qa_logger is None:
        _qa_logger = QACollaborationLogger(log_dir)
    return _qa_logger
