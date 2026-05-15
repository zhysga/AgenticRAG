"""
多模型LLM客户端
支持deepseek、doubao、qwen三大模型API的回退机制
"""

import logging
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
from openai import OpenAI
from llama_index.core.llms import LLM, CompletionResponse, CompletionResponseGen, ChatResponse
from llama_index.core.llms.llm import MessageRole, ChatMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MultiLLMClient(LLM):
    """
    支持三个API回退机制的LLM客户端
    优先级: deepseek → doubao → qwen
    """
    
    # 允许额外字段，避免 Pydantic 验证错误
    model_config = {"extra": "allow"}
    
    def __init__(self, project_dir: str, llm: str = "fallback", **kwargs):
        # 调用父类初始化，传入必要的参数
        super().__init__(**kwargs)
        
        self.project_dir = project_dir
        
        # 设置日志目录
        self.log_dir = Path(project_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建详细日志文件（固定文件名，便于追踪）
        self.log_file = self.log_dir / "llm_api_calls.jsonl"
        
        # 初始化调用计数器
        self.call_counter = 0
        
        # 记录初始化信息
        logger.info(f"MultiLLMClient日志文件: {self.log_file}")
        
        # API配置
        self.api_configs = {
            "doubao": {
                "api_key": os.environ.get("DOUBAO_API_KEY", "a34b7b1a-2660-4ba4-972f-3ebadafbf14c"),
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "model": "doubao-seed-1-6-flash-250615",
                "proxy": None
            },
            "deepseek": {
                "api_key": os.environ.get("DEEPSEEK_API_KEY", "sk-8697d833c5804a17b5320892daa0fbde"),
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "proxy": None
            },
            "qwen": {
                "api_key": os.environ.get("QWEN_API_KEY", ""),
                "base_url": "https://api.qwen.com/v1",
                "model": "qwen-7b",
                "proxy": None
            }
        }
        
        # 初始化客户端
        self.clients = {}
        for name, config in self.api_configs.items():
            if not config["api_key"] or config["api_key"] == "":
                logger.warning(f"{name} API key not configured, skipping")
                continue
                
            http_client = None
            if config.get("proxy"):
                http_client = httpx.Client(proxy=config["proxy"])
            
            try:
                self.clients[name] = OpenAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                    timeout=300,
                    http_client=http_client
                )
                logger.info(f"Successfully initialized {name} client")
            except Exception as e:
                logger.error(f"Failed to initialize {name} client: {e}")
                continue  # 继续尝试其他客户端
        
        # 回退顺序 - 优先deepseek，然后doubao，最后qwen
        self.fallback_order = [name for name in ["deepseek", "doubao", "qwen"] 
                              if name in self.clients]
        
        if not self.fallback_order:
            raise ValueError("No valid LLM clients configured")
        
        logger.info(f"Initialized MultiLLMClient with fallback order: {self.fallback_order}")
    
    def _log_llm_call(self, call_data: Dict[str, Any]):
        """记录LLM调用的详细信息"""
        try:
            def _json_default(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(call_data, f, ensure_ascii=False, default=_json_default)
                f.write('\n')
        except Exception as e:
            logger.warning(f"写入LLM调用日志失败: {e}")
    
    def _generate_single(self, prompt: str, **kwargs) -> str:
        """
        单个prompt的生成，支持回退机制，并记录详细日志
        """
        self.call_counter += 1
        call_start_time = datetime.now()
        
        # 准备日志数据结构
        log_data = {
            "call_id": self.call_counter,
            "timestamp": call_start_time.isoformat(),
            "prompt": prompt,
            "prompt_length": len(prompt),
            "kwargs": kwargs,
            "attempts": []
        }
        
        for api_name in self.fallback_order:
            attempt_start = datetime.now()
            attempt_data = {
                "api_name": api_name,
                "model": self.api_configs[api_name]["model"],
                "attempt_time": attempt_start.isoformat(),
                "success": False,
                "error": None,
                "response": None,
                "duration_ms": 0
            }
            
            try:
                logger.info(f"[调用#{self.call_counter}] 尝试使用 {api_name} API")
                client = self.clients[api_name]
                config = self.api_configs[api_name]
                
                # 准备请求参数
                request_params = {
                    "model": config["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 2048),
                    "temperature": kwargs.get("temperature", 0.7),
                }
                
                # 添加其他参数
                for k, v in kwargs.items():
                    if k not in ["max_tokens", "temperature"]:
                        request_params[k] = v
                
                # 记录请求参数
                attempt_data["request_params"] = request_params.copy()
                
                # 调用API
                response = client.chat.completions.create(**request_params)
                
                # 计算耗时
                attempt_end = datetime.now()
                attempt_data["duration_ms"] = (attempt_end - attempt_start).total_seconds() * 1000
                
                # 提取响应内容
                message = response.choices[0].message
                text = message.content or ""
                
                attempt_data.update({
                    "success": True,
                    "response": {
                        "content": text,
                        "content_length": len(text),
                        "finish_reason": response.choices[0].finish_reason,
                        "usage": response.usage.model_dump() if response.usage else None
                    }
                })
                
                log_data["attempts"].append(attempt_data)
                log_data["final_result"] = {
                    "success": True,
                    "used_api": api_name,
                    "total_duration_ms": (attempt_end - call_start_time).total_seconds() * 1000
                }
                
                self._log_llm_call(log_data)
                
                logger.info(f"[调用#{self.call_counter}] {api_name} API调用成功，耗时: {attempt_data['duration_ms']:.2f}ms")
                return text
                
            except Exception as e:
                attempt_end = datetime.now()
                attempt_data["duration_ms"] = (attempt_end - attempt_start).total_seconds() * 1000
                attempt_data["error"] = str(e)
                
                log_data["attempts"].append(attempt_data)
                
                logger.warning(f"[调用#{self.call_counter}] {api_name} API调用失败: {e}, 耗时: {attempt_data['duration_ms']:.2f}ms")
                
                if api_name == self.fallback_order[-1]:
                    final_end = datetime.now()
                    log_data["final_result"] = {
                        "success": False,
                        "error": "所有API都失败了",
                        "total_duration_ms": (final_end - call_start_time).total_seconds() * 1000
                    }
                    self._log_llm_call(log_data)
                    logger.error(f"[调用#{self.call_counter}] 所有API都失败了")
                    return "抱歉，当前无法生成回答。"
                continue
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """实现llama_index的complete接口"""
        text = self._generate_single(prompt, **kwargs)
        return CompletionResponse(text=text)
    
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """实现llama_index的stream_complete接口"""
        text = self._generate_single(prompt, **kwargs)
        yield CompletionResponse(text=text)
    
    def chat(self, messages, **kwargs) -> ChatResponse:
        """实现chat接口，返回ChatResponse以兼容llama_index内部调用"""
        # 将消息转换为prompt
        if isinstance(messages, list) and len(messages) > 0:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                prompt = last_message.content
            else:
                prompt = str(last_message)
        else:
            prompt = str(messages)

        text = self._generate_single(prompt, **kwargs)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))
    
    def stream_chat(self, messages, **kwargs):
        """实现stream_chat接口；简化为一次性返回以保持兼容"""
        resp = self.chat(messages, **kwargs)
        yield resp
    
    def acomplete(self, prompt: str, **kwargs):
        """异步complete接口"""
        return asyncio.run(self.complete(prompt, **kwargs))
    
    def achat(self, messages, **kwargs):
        """异步chat接口"""
        return asyncio.run(self.chat(messages, **kwargs))
    
    def astream_complete(self, prompt: str, **kwargs):
        """异步stream_complete接口"""
        return asyncio.run(self.stream_complete(prompt, **kwargs))
    
    def astream_chat(self, messages, **kwargs):
        """异步stream_chat接口"""
        return asyncio.run(self.stream_chat(messages, **kwargs))
    
    @property
    def metadata(self):
        """返回模型元数据（对象而非dict，兼容属性访问）"""
        class _LLMMetadata:
            def __init__(self):
                self.model_name = "multi_llm_client"
                self.is_chat_model = True
                self.is_function_calling_model = False
                self.context_window = 8192
                self.num_output = 2048
                self.model_family = "multi"

            def __repr__(self) -> str:
                return (
                    f"_LLMMetadata(model_name={self.model_name}, "
                    f"is_chat_model={self.is_chat_model}, "
                    f"is_function_calling_model={self.is_function_calling_model}, "
                    f"context_window={self.context_window}, num_output={self.num_output}, "
                    f"model_family={self.model_family})"
                )

        return _LLMMetadata()
    
    @property
    def context_window(self):
        """返回上下文窗口大小"""
        return 8192
    
    @property 
    def model_name(self):
        """返回模型名称"""
        return "multi_llm_client"
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """获取调用统计信息"""
        if not self.log_file.exists():
            return {"total_calls": 0, "api_usage": {}}
        
        stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "api_usage": {},
            "average_duration_ms": 0
        }
        
        total_duration = 0
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        call_data = json.loads(line)
                        stats["total_calls"] += 1
                        
                        if call_data.get("final_result", {}).get("success", False):
                            stats["successful_calls"] += 1
                            used_api = call_data["final_result"]["used_api"]
                            stats["api_usage"][used_api] = stats["api_usage"].get(used_api, 0) + 1
                        else:
                            stats["failed_calls"] += 1
                        
                        if "total_duration_ms" in call_data.get("final_result", {}):
                            total_duration += call_data["final_result"]["total_duration_ms"]
        
        except Exception as e:
            logger.warning(f"读取调用统计失败: {e}")
        
        if stats["total_calls"] > 0:
            stats["average_duration_ms"] = total_duration / stats["total_calls"]
        
        return stats 