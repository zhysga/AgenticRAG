"""
API客户端封装 - 前端与后端通信的核心桥梁

这个类封装了所有与后端API的通信逻辑，是前后端结合的关键组件。
前端通过这个客户端调用后端的RESTful API，实现数据交互和业务功能。

前后端结合机制：
1. 使用requests库发起HTTP请求
2. 统一的错误处理和响应解析
3. 会话管理保持连接状态
4. 支持认证和授权
5. 提供业务方法封装，简化前端调用

设计模式：
- 单例模式：每个前端实例使用一个APIClient
- 门面模式：封装复杂API调用为简单方法
- 适配器模式：统一不同API的调用方式
"""

import requests  # HTTP请求库，用于与后端API通信
import json  # JSON数据处理
from typing import Dict, Any, List, Optional  # 类型注解
import logging  # 日志记录
import time

# 创建日志记录器
logger = logging.getLogger(__name__)


class APIClient:
    """
    API客户端类 - 封装所有后端API调用
    
    这个类负责：
    1. 管理HTTP会话和连接
    2. 处理请求和响应
    3. 错误处理和重试逻辑
    4. 提供业务方法接口
    
    前后端通信流程：
    前端UI → APIClient → HTTP请求 → 后端FastAPI → 业务处理 → 返回响应
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "", timeout: float = 300.0):
        """
        初始化API客户端
        
        Args:
            base_url: 后端API的基础URL，默认指向本地开发环境
            api_key: API密钥，用于认证（可选）
        """
        # 清理URL末尾的斜杠，确保URL格式正确
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # 创建HTTP会话，保持连接复用，提高性能
        self.session = requests.Session()
        
        # 设置默认请求头 - 告诉后端我们发送的是JSON数据
        self.session.headers.update({
            "Content-Type": "application/json",  # 内容类型为JSON
            "User-Agent": "LangGraph-RAG-Frontend/1.0"  # 用户代理标识
        })
        
        # 如果有API密钥，添加认证头
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}"  # Bearer Token认证
            })
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """发起HTTP请求"""
        try:
            url = f"{self.base_url}{endpoint}"
            _timeout = timeout if timeout is not None else self.timeout
            hdrs = dict(self.session.headers)
            if headers:
                hdrs.update(headers)
            start = time.time()
            self._log_request_start(method, url, params, data, _timeout, hdrs)

            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=_timeout, headers=hdrs)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, params=params, timeout=_timeout, headers=hdrs)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, params=params, timeout=_timeout, headers=hdrs)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, params=params, timeout=_timeout, headers=hdrs)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            duration_ms = (time.time() - start) * 1000
            self._log_request_end(method, url, response.status_code, duration_ms)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return {
                "status": "error",
                "message": f"请求失败: {str(e)}",
                "request_id": ""
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return {
                "status": "error",
                "message": f"响应解析失败: {str(e)}",
                "request_id": ""
            }
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return {
                "status": "error",
                "message": f"未知错误: {str(e)}",
                "request_id": ""
            }

    def _log_request_start(self, method: str, url: str, params: Optional[Dict[str, Any]], data: Optional[Dict[str, Any]], timeout: float, headers: Dict[str, str]):
        try:
            logger.info(
                "[frontend] api_client: request",
                extra={
                    "method": method,
                    "url": url,
                    "timeout": timeout,
                    "params_keys": list((params or {}).keys()),
                    "body_keys": list((data or {}).keys()),
                    "x_request_id": headers.get("X-Request-ID"),
                },
            )
        except Exception:
            pass

    def _log_request_end(self, method: str, url: str, status_code: int, duration_ms: float):
        try:
            logger.info(
                "[frontend] api_client: response",
                extra={
                    "method": method,
                    "url": url,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
            )
        except Exception:
            pass
    
    # 健康检查
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return self._make_request("GET", "/auth/health")
    
    # 知识库管理API
    def create_knowledge_base(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建知识库"""
        return self._make_request("POST", "/kb/create", data=data)
    
    def list_knowledge_bases(
        self,
        page: int = 1,
        size: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """列出知识库"""
        params = {"page": page, "size": size}
        if filters:
            params.update(filters)
        return self._make_request("GET", "/kb/list", params=params)
    
    def upload_files(self, data: Dict[str, Any], files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """上传文件"""
        # 将文件数据添加到请求中
        data["files"] = files
        return self._make_request("POST", "/kb/upload", data=data)
    
    def get_knowledge_base_files(
        self,
        kb_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取知识库文件"""
        params = {"kb_id": kb_id}
        if filters:
            params.update(filters)
        return self._make_request("GET", "/kb/files", params=params)
    
    def reindex_knowledge_base(self, kb_id: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """重新索引知识库（支持后端accepted异步语义）"""
        data = {"kb_id": kb_id, "file_ids": file_ids}
        # 后端为异步后台任务，接口应快速返回
        resp = self._make_request("POST", "/kb/reindex", data=data, timeout=(3.0, 5.0))
        # 兼容后端异步：accepted 也视为受理成功，避免前端误判为失败
        if isinstance(resp, dict) and resp.get("status") in ("accepted", "success"):
            normalized = dict(resp)
            # 规范化：将 accepted 归一为 success，便于上层统一处理
            normalized["status"] = "success"
            if resp.get("status") == "accepted" and not normalized.get("message"):
                normalized["message"] = "重新索引任务已受理，正在后台执行"
            return normalized
        return resp

    def get_reindex_status(self, kb_id: str) -> Dict[str, Any]:
        """获取重新索引状态"""
        endpoint = f"/kb/reindex/status/{kb_id}"
        return self._make_request("GET", endpoint, timeout=(3.0, 15.0))
    
    def delete_knowledge_base(self, kb_id: str, confirm: bool = False) -> Dict[str, Any]:
        """删除知识库"""
        data = {"kb_id": kb_id, "confirm": confirm}
        return self._make_request("POST", "/kb/delete", data=data)
    
    # 智能体管理API
    def create_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建智能体"""
        return self._make_request("POST", "/agent/create", data=data)
    
    def list_agents(
        self,
        page: int = 1,
        size: int = 10,
        agent_type: Optional[str] = None,
        routing_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """列出智能体"""
        params = {"page": page, "size": size}
        if agent_type:
            params["agent_type"] = agent_type
        if routing_tags:
            # 后端期望以逗号分隔的字符串形式提交路由标签
            params["routing_tags"] = ",".join(routing_tags)
        return self._make_request("GET", "/agent/list", params=params)
    
    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体信息"""
        endpoint = f"/agent/get/{agent_id}"
        return self._make_request("GET", endpoint)
    
    def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """更新智能体"""
        data = {"agent_id": agent_id, "updates": updates}
        return self._make_request("POST", "/agent/update", data=data)
    
    def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """删除智能体"""
        data = {"agent_id": agent_id}
        return self._make_request("POST", "/agent/delete", data=data)
    
    def test_agent(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """测试智能体"""
        return self._make_request("POST", "/agent/test", data=data)
    
    # 问答协作API
    def ask_question(self, data: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """提问 - 使用较长超时，因为工作流涉及多智能体串行调用 LLM"""
        headers = {"X-Request-ID": request_id} if request_id else None
        # 工作流可能执行 analyst → writer → critic → synthesizer 多轮 LLM 调用
        # 每轮 LLM 可能需要 10-30 秒，总耗时可达 120-300 秒
        return self._make_request("POST", "/chat/ask", data=data, headers=headers, timeout=300.0)
    
    def get_chat_history(
        self,
        session_id: str,
        page: int = 1,
        size: int = 10
    ) -> Dict[str, Any]:
        """获取聊天历史"""
        params = {
            "session_id": session_id,
            "page": page,
            "size": size
        }
        return self._make_request("GET", "/chat/history", params=params)
    
    def list_chat_sessions(
        self,
        page: int = 1,
        size: int = 10,
        agent_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """列出聊天会话"""
        params = {"page": page, "size": size}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status
        return self._make_request("GET", "/chat/sessions", params=params)
    
    def delete_chat_session(self, session_id: str) -> Dict[str, Any]:
        """删除聊天会话"""
        data = {"session_id": session_id}
        return self._make_request("POST", "/chat/delete_session", data=data)
    
    # 流式问答API
    def ask_question_stream(self, data: Dict[str, Any]):
        """流式提问"""
        try:
            url = f"{self.base_url}/chat/ask"
            data["stream"] = True
            
            response = self.session.post(url, json=data, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line.decode('utf-8'))
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"流式请求失败: {e}")
            yield {
                "status": "error",
                "message": f"流式请求失败: {str(e)}",
                "request_id": ""
            }
    
    # 批量操作API
    def batch_create_knowledge_bases(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量创建知识库"""
        data = {"knowledge_bases": data_list}
        return self._make_request("POST", "/kb/batch_create", data=data)
    
    def batch_create_agents(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量创建智能体"""
        data = {"agents": data_list}
        return self._make_request("POST", "/agent/batch_create", data=data)
    
    # 统计信息API
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return self._make_request("GET", "/stats/system")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        return self._make_request("GET", "/stats/agents")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self._make_request("GET", "/stats/knowledge_bases")
    
    # 配置管理API
    def get_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self._make_request("GET", "/config/get")
    
    def update_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新系统配置"""
        return self._make_request("POST", "/config/update", data=config_data)
    
    # 工具方法
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self.api_key = api_key
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })
        else:
            self.session.headers.pop("Authorization", None)
    
    def set_base_url(self, base_url: str):
        """设置基础URL"""
        self.base_url = base_url.rstrip("/")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        try:
            health_response = self.health_check()
            return {
                "connected": health_response.get("status") == "success",
                "base_url": self.base_url,
                "api_key_set": bool(self.api_key),
                "response": health_response
            }
        except Exception as e:
            return {
                "connected": False,
                "base_url": self.base_url,
                "api_key_set": bool(self.api_key),
                "error": str(e)
            }
