"""
数据存储适配器
支持智能体和知识库的CRUD操作
"""
import logging
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import uuid
from enum import Enum

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class StorageAdapter:
    """数据存储适配器"""
    
    def __init__(self, storage_dir: str = "./storage/data"):
        self.storage_dir = storage_dir
        self.agents_file = os.path.join(storage_dir, "agents.json")
        self.knowledge_bases_file = os.path.join(storage_dir, "knowledge_bases.json")
        self.kb_files_file = os.path.join(storage_dir, "kb_files.json")
        self._initialize_storage()
    
    def _safe_json_load(self, file_path: str) -> Dict[str, Any]:
        """安全加载JSON：空文件/无效JSON时返回空字典并记录调试日志而非错误。
        
        说明：
        - 某些情况下外部工具可能创建了空文件，直接 json.load 会抛出 'Expecting value'
        - 这里统一兜底，避免影响上层流程
        """
        try:
            if not os.path.exists(file_path):
                return {}
            with open(file_path, 'r', encoding='utf-8') as f:
                raw = f.read()
                if raw is None:
                    return {}
                raw_stripped = raw.strip()
                if raw_stripped == "":
                    logger.warning(f"JSON文件为空，返回默认值: {file_path}")
                    return {}
                try:
                    return json.loads(raw_stripped)
                except Exception as parse_err:
                    logger.warning(f"JSON解析失败，返回默认值: {file_path}，错误: {parse_err}")
                    return {}
        except Exception as e:
            logger.error(f"安全加载JSON失败: {file_path}，错误: {e}")
            return {}
    
    def _initialize_storage(self):
        """初始化存储目录和文件"""
        try:
            # 确保存储目录存在
            os.makedirs(self.storage_dir, exist_ok=True)
            
            def _json_default(obj):
                # 统一处理不可序列化对象
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if isinstance(obj, Enum):
                    return obj.value
                return str(obj)
            
            # 初始化智能体文件
            if not os.path.exists(self.agents_file):
                with open(self.agents_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=2, default=_json_default)
            
            # 初始化知识库文件
            if not os.path.exists(self.knowledge_bases_file):
                with open(self.knowledge_bases_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=2, default=_json_default)

            # 初始化知识库文件明细
            if not os.path.exists(self.kb_files_file):
                with open(self.kb_files_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=2, default=_json_default)
            
            logger.info(f"存储适配器初始化成功: {self.storage_dir}")
            
        except Exception as e:
            logger.error(f"存储适配器初始化失败: {e}")
            raise
    
    def _load_agents(self) -> Dict[str, Dict[str, Any]]:
        """加载智能体数据"""
        try:
            return self._safe_json_load(self.agents_file)
        except Exception as e:
            logger.error(f"加载智能体数据失败: {e}")
            return {}
    
    def _save_agents(self, agents: Dict[str, Dict[str, Any]]):
        """保存智能体数据"""
        try:
            def _json_default(obj):
                # 统一处理不可序列化对象
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if isinstance(obj, Enum):
                    return obj.value
                return str(obj)

            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump(agents, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as e:
            logger.error(f"保存智能体数据失败: {e}")
            raise
    
    def _load_knowledge_bases(self) -> Dict[str, Dict[str, Any]]:
        """加载知识库数据"""
        try:
            return self._safe_json_load(self.knowledge_bases_file)
        except Exception as e:
            logger.error(f"加载知识库数据失败: {e}")
            return {}

    def _load_kb_files(self) -> Dict[str, Dict[str, Any]]:
        """加载知识库文件数据"""
        try:
            return self._safe_json_load(self.kb_files_file)
        except Exception as e:
            logger.error(f"加载知识库文件数据失败: {e}")
            return {}
    
    def _save_knowledge_bases(self, knowledge_bases: Dict[str, Dict[str, Any]]):
        """保存知识库数据"""
        try:
            def _json_default(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if isinstance(obj, Enum):
                    return obj.value
                return str(obj)

            with open(self.knowledge_bases_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_bases, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as e:
            logger.error(f"保存知识库数据失败: {e}")
            raise

    def _save_kb_files(self, kb_files: Dict[str, Dict[str, Any]]):
        """保存知识库文件数据"""
        try:
            def _json_default(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if isinstance(obj, Enum):
                    return obj.value
                return str(obj)

            with open(self.kb_files_file, 'w', encoding='utf-8') as f:
                json.dump(kb_files, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as e:
            logger.error(f"保存知识库文件数据失败: {e}")
            raise
    
    # 智能体相关方法
    def save_agent(self, agent_data: Dict[str, Any]) -> bool:
        """保存智能体"""
        try:
            agents = self._load_agents()
            agent_id = agent_data.get("agent_id")
            
            if not agent_id:
                logger.error("智能体数据缺少agent_id")
                return False
            
            # 保存智能体数据
            agents[agent_id] = agent_data
            self._save_agents(agents)
            
            logger.info(f"智能体保存成功: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存智能体失败: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体"""
        try:
            agents = self._load_agents()
            return agents.get(agent_id)
        except Exception as e:
            logger.error(f"获取智能体失败: {e}")
            return None
    
    def list_agents(
        self, 
        page: int = 1, 
        size: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """列出智能体"""
        try:
            agents = self._load_agents()
            agents_list = list(agents.values())
            
            # 应用过滤条件
            if filters:
                filtered_agents = []
                for agent in agents_list:
                    match = True
                    for key, value in filters.items():
                        if agent.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_agents.append(agent)
                agents_list = filtered_agents
            
            # 分页处理
            start_index = (page - 1) * size
            end_index = start_index + size
            
            logger.info(f"列出智能体: 总数={len(agents_list)}, 分页={page}/{size}")
            return agents_list[start_index:end_index]
            
        except Exception as e:
            logger.error(f"列出智能体失败: {e}")
            return []
    
    def update_agent(self, agent_id: str, update_data: Dict[str, Any]) -> bool:
        """更新智能体"""
        try:
            agents = self._load_agents()
            
            if agent_id not in agents:
                logger.warning(f"智能体不存在: {agent_id}")
                return False
            
            # 更新数据
            agents[agent_id].update(update_data)
            self._save_agents(agents)
            
            logger.info(f"智能体更新成功: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新智能体失败: {e}")
            return False
    
    def delete_agent(self, agent_id: str) -> bool:
        """删除智能体"""
        try:
            agents = self._load_agents()
            
            if agent_id not in agents:
                logger.warning(f"智能体不存在: {agent_id}")
                return False
            
            # 删除智能体
            del agents[agent_id]
            self._save_agents(agents)
            
            logger.info(f"智能体删除成功: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除智能体失败: {e}")
            return False
    
    # 知识库相关方法
    def save_knowledge_base(self, kb_data: Dict[str, Any]) -> bool:
        """保存知识库"""
        try:
            knowledge_bases = self._load_knowledge_bases()
            kb_id = kb_data.get("kb_id")
            
            if not kb_id:
                logger.error("知识库数据缺少kb_id")
                return False
            
            # 保存知识库数据
            knowledge_bases[kb_id] = kb_data
            self._save_knowledge_bases(knowledge_bases)
            
            logger.info(f"知识库保存成功: {kb_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")
            return False
    
    def get_knowledge_base(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """获取知识库"""
        try:
            knowledge_bases = self._load_knowledge_bases()
            return knowledge_bases.get(kb_id)
        except Exception as e:
            logger.error(f"获取知识库失败: {e}")
            return None
    
    def list_knowledge_bases(
        self, 
        page: int = 1, 
        size: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """列出知识库"""
        try:
            knowledge_bases = self._load_knowledge_bases()
            kb_list = list(knowledge_bases.values())
            
            # 应用过滤条件
            if filters:
                filtered_kbs = []
                for kb in kb_list:
                    match = True
                    for key, value in filters.items():
                        if kb.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_kbs.append(kb)
                kb_list = filtered_kbs
            
            # 分页处理
            start_index = (page - 1) * size
            end_index = start_index + size
            
            logger.info(f"列出知识库: 总数={len(kb_list)}, 分页={page}/{size}")
            return kb_list[start_index:end_index]
            
        except Exception as e:
            logger.error(f"列出知识库失败: {e}")
            return []
    
    def update_knowledge_base(self, kb_id: str, update_data: Dict[str, Any]) -> bool:
        """更新知识库"""
        try:
            knowledge_bases = self._load_knowledge_bases()
            
            if kb_id not in knowledge_bases:
                logger.warning(f"知识库不存在: {kb_id}")
                return False
            
            # 更新数据
            knowledge_bases[kb_id].update(update_data)
            self._save_knowledge_bases(knowledge_bases)
            
            logger.info(f"知识库更新成功: {kb_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新知识库失败: {e}")
            return False
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库"""
        try:
            knowledge_bases = self._load_knowledge_bases()
            
            if kb_id not in knowledge_bases:
                logger.warning(f"知识库不存在: {kb_id}")
                return False
            
            # 删除知识库
            del knowledge_bases[kb_id]
            self._save_knowledge_bases(knowledge_bases)
            
            logger.info(f"知识库删除成功: {kb_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除知识库失败: {e}")
            return False

    def delete_kb_files(self, kb_id: str) -> bool:
        """删除指定知识库的所有文件元数据（kb_files.json 中对应条目）"""
        try:
            kb_files = self._load_kb_files()
            if kb_id in kb_files:
                del kb_files[kb_id]
                self._save_kb_files(kb_files)
                logger.info(f"已清理知识库文件元数据: {kb_id}")
                return True
            logger.info(f"未找到需要清理的知识库文件元数据: {kb_id}")
            return False
        except Exception as e:
            logger.error(f"清理知识库文件元数据失败: {e}")
            return False

    # 文件相关方法
    def add_files(
        self,
        kb_id: str,
        files: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        auto_index: bool = True
    ) -> List[Dict[str, Any]]:
        """向知识库添加文件并持久化"""
        try:
            kb_files = self._load_kb_files()
            now = datetime.utcnow()
            # 计算上传文件保存目录: <project_root>/storage/uploads/<kb_id>
            base_storage_dir = os.path.dirname(self.storage_dir)  # 默认 ./storage
            uploads_root = os.path.join(base_storage_dir, "uploads")
            kb_upload_dir = os.path.join(uploads_root, kb_id)
            os.makedirs(kb_upload_dir, exist_ok=True)

            # 确保知识库文件容器存在
            if kb_id not in kb_files:
                kb_files[kb_id] = {}

            saved_files: List[Dict[str, Any]] = []
            for f in files:
                file_id = f.get("file_id") or str(uuid.uuid4())
                file_name = f.get("file_name") or "unknown"
                file_type = f.get("file_type") or "text/plain"
                labels = f.get("labels") or []
                content = f.get("content")
                file_size = f.get("file_size")
                if file_size is None and isinstance(content, str):
                    file_size = len(content)
                elif file_size is None and content is not None:
                    try:
                        file_size = len(content)
                    except Exception:
                        file_size = 0

                # 将内容持久化到 storage/uploads/<kb_id>/<file_id>_<file_name>
                saved_path = None
                try:
                    # 规范文件名，避免非法字符
                    safe_name = str(file_name).replace("\\", "_").replace("/", "_")
                    target_path = os.path.join(kb_upload_dir, f"{file_id}_{safe_name}")

                    if isinstance(content, (str, bytes)):
                        # 直接写入内容
                        saved_path = target_path
                        if isinstance(content, bytes):
                            with open(saved_path, "wb") as wf:
                                wf.write(content)
                        else:
                            with open(saved_path, "w", encoding="utf-8") as wf:
                                wf.write(content)
                    elif content is None:
                        # 前端未传内容，尝试从原始文件路径复制
                        source_path = f.get("file_path")
                        if source_path and os.path.exists(source_path):
                            import shutil
                            shutil.copy2(source_path, target_path)
                            saved_path = target_path
                            logger.info(f"从原始路径复制文件到存储目录: {source_path} -> {target_path}")
                            # 复制成功后更新 file_size
                            try:
                                file_size = os.path.getsize(target_path)
                            except Exception:
                                pass
                        else:
                            logger.warning(f"文件无内容且原始路径不可用: file_id={file_id}, file_path={source_path}")
                    else:
                        logger.warning(f"文件内容类型不受支持，跳过持久化到磁盘: type={type(content)}")
                except Exception as write_err:
                    logger.warning(f"保存文件到磁盘失败（继续保存元数据）: kb_id={kb_id}, file_id={file_id}, err={write_err}")

                file_record = {
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "file_size": int(file_size or 0),
                    "labels": labels,
                    # 将保存到磁盘的绝对路径写入，便于后续索引器直接读取
                    "file_path": os.path.abspath(saved_path) if saved_path else None,
                    "chunk_count": 0,
                    "upload_time": now.isoformat(),
                    "index_status": "pending",
                    "metadata": {
                        "kb_id": kb_id,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "auto_index": auto_index,
                        "user_id": f.get("user_id"),
                        # 为简化演示，原始内容暂存入元数据
                        "content": content,
                    },
                }

                # 保存到映射
                kb_files[kb_id][file_id] = file_record
                saved_files.append(file_record)

            # 持久化
            self._save_kb_files(kb_files)
            logger.info(f"知识库{kb_id}新增文件: {len(saved_files)}")
            return saved_files

        except Exception as e:
            logger.error(f"添加文件失败: {e}")
            return []

    def list_files(
        self,
        page: int = 1,
        size: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """列出文件，支持按kb_id、file_type、labels过滤并分页"""
        try:
            kb_files = self._load_kb_files()
            kb_id = filters.get("kb_id") if filters else None
            file_type = filters.get("file_type") if filters else None
            labels = filters.get("labels") if filters else None

            all_files: List[Dict[str, Any]] = []
            if kb_id:
                kb_map = kb_files.get(kb_id, {})
                all_files = list(kb_map.values())
            else:
                # 未指定kb_id则列出所有（通常不建议，但用于通用性）
                for _kb_id, kb_map in kb_files.items():
                    all_files.extend(list(kb_map.values()))

            # 过滤类型
            if file_type:
                all_files = [f for f in all_files if f.get("file_type") == file_type]

            # 过滤标签（至少有一个标签匹配则保留）
            if labels:
                def labels_match(flabels: List[str], target: List[str]) -> bool:
                    return any(l in flabels for l in target)
                all_files = [f for f in all_files if labels_match(f.get("labels", []), labels)]

            # 分页
            start_index = (page - 1) * size
            end_index = start_index + size

            logger.info(f"列出文件: kb_id={kb_id}, 总数={len(all_files)}, 分页={page}/{size}")
            return all_files[start_index:end_index]

        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    def update_file_status(
        self,
        kb_id: str,
        file_id: str,
        index_status: str = "completed",
        chunk_count: int = 0
    ) -> bool:
        """更新文件索引状态和分块数"""
        try:
            kb_files = self._load_kb_files()
            
            if kb_id not in kb_files:
                logger.warning(f"知识库 {kb_id} 不存在")
                return False
            
            if file_id not in kb_files[kb_id]:
                logger.warning(f"文件 {file_id} 不存在")
                return False
            
            # 更新文件状态
            kb_files[kb_id][file_id]["index_status"] = index_status
            kb_files[kb_id][file_id]["chunk_count"] = chunk_count
            
            # 持久化
            self._save_kb_files(kb_files)
            logger.info(f"更新文件状态: kb_id={kb_id}, file_id={file_id}, status={index_status}, chunks={chunk_count}")
            return True
            
        except Exception as e:
            logger.error(f"更新文件状态失败: {e}")
            return False
    
    def update_files_status(
        self,
        kb_id: str,
        file_status_map: Dict[str, Dict[str, Any]]
    ) -> bool:
        """批量更新文件索引状态和分块数
        
        Args:
            kb_id: 知识库ID
            file_status_map: 文件状态映射，格式为 {file_id: {"index_status": "completed", "chunk_count": 10}}
        """
        try:
            kb_files = self._load_kb_files()
            
            if kb_id not in kb_files:
                logger.warning(f"知识库 {kb_id} 不存在")
                return False
            
            updated_count = 0
            for file_id, status_info in file_status_map.items():
                if file_id in kb_files[kb_id]:
                    kb_files[kb_id][file_id]["index_status"] = status_info.get("index_status", "completed")
                    kb_files[kb_id][file_id]["chunk_count"] = status_info.get("chunk_count", 0)
                    updated_count += 1
                else:
                    logger.warning(f"文件 {file_id} 不存在，跳过更新")
            
            if updated_count > 0:
                # 持久化
                self._save_kb_files(kb_files)
                logger.info(f"批量更新文件状态: kb_id={kb_id}, 更新了 {updated_count} 个文件")
                return True
            else:
                logger.warning(f"没有文件需要更新")
                return False
                
        except Exception as e:
            logger.error(f"批量更新文件状态失败: {e}")
            return False