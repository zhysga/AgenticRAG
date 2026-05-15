"""
智能体管理服务
"""
import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from backend.models.agent import AgentInfo, AgentCreate, AgentType
from backend.adapters.storage_adapter import StorageAdapter
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AgentService:
    """智能体管理服务"""
    
    def __init__(self, storage: StorageAdapter):
        self.storage = storage
        self._agents_cache = {}  # 智能体缓存
    
    def create_agent(self, agent_data: AgentCreate) -> AgentInfo:
        """创建智能体"""
        try:
            logger.info(f"开始创建智能体: {agent_data.name}")
            
            # 生成智能体ID
            agent_id = str(uuid.uuid4())
            
            # 创建智能体信息
            agent_info = AgentInfo(
                agent_id=agent_id,
                name=agent_data.name,
                description=agent_data.description,
                agent_type=agent_data.agent_type,
                system_prompt=agent_data.system_prompt,
                tools=agent_data.tools,
                bind_kb_ids=agent_data.bind_kb_ids,
                routing_tags=agent_data.routing_tags,
                max_turns=agent_data.max_turns,
                temperature=agent_data.temperature,
                model_config=agent_data.model_config,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status="active"
            )
            
            # 存储到数据库
            self.storage.save_agent(agent_info.dict())
            
            # 更新缓存
            self._agents_cache[agent_id] = agent_info
            
            logger.info(f"智能体创建成功: {agent_id}")
            return agent_info
            
        except Exception as e:
            logger.error(f"创建智能体失败: {e}")
            raise
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """获取智能体信息（按ID）"""
        try:
            # 先检查缓存
            if agent_id in self._agents_cache:
                return self._agents_cache[agent_id]
            
            # 从数据库获取
            agent_data = self.storage.get_agent(agent_id)
            if agent_data:
                agent_info = AgentInfo(**agent_data)
                self._agents_cache[agent_id] = agent_info
                return agent_info
            
            return None
            
        except Exception as e:
            logger.error(f"获取智能体失败: {e}")
            return None
    
    def get_agent_by_name(self, name: str) -> Optional[AgentInfo]:
        """按名称获取智能体信息"""
        try:
            # 从存储中获取所有智能体
            agents_data = self.storage.list_agents(page=1, size=1000)
            
            # 查找匹配名称的智能体
            for agent_data in agents_data:
                if agent_data.get("name") == name:
                    agent_info = AgentInfo(**agent_data)
                    # 更新缓存
                    self._agents_cache[agent_info.agent_id] = agent_info
                    return agent_info
            
            return None
            
        except Exception as e:
            logger.error(f"按名称获取智能体失败: {e}")
            return None
    
    def list_agents(
        self,
        page: int = 1,
        size: int = 10,
        agent_type: Optional[AgentType] = None,
        routing_tags: Optional[List[str]] = None
    ) -> List[AgentInfo]:
        """列出智能体"""
        try:
            logger.info(f"列出智能体: page={page}, size={size}")
            
            # 构建过滤条件
            filters = {}
            if agent_type:
                filters["agent_type"] = agent_type.value
            if routing_tags:
                filters["routing_tags"] = routing_tags
            
            # 从数据库获取
            agents_data = self.storage.list_agents(
                page=page,
                size=size,
                filters=filters
            )
            
            # 转换为AgentInfo对象
            agents = []
            for agent_data in agents_data:
                agent_info = AgentInfo(**agent_data)
                agents.append(agent_info)
            
            logger.info(f"获取到{len(agents)}个智能体")
            return agents
            
        except Exception as e:
            logger.error(f"列出智能体失败: {e}")
            return []
    
    def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Optional[AgentInfo]:
        """更新智能体"""
        try:
            logger.info(f"开始更新智能体: {agent_id}")
            
            # 获取现有智能体
            agent_info = self.get_agent(agent_id)
            if not agent_info:
                logger.warning(f"智能体不存在: {agent_id}")
                return None
            
            # 更新字段
            update_data = agent_info.dict()
            for key, value in updates.items():
                if key in update_data:
                    update_data[key] = value
            
            # 更新时间戳
            update_data["updated_at"] = datetime.now()
            
            # 保存到数据库
            self.storage.update_agent(agent_id, update_data)
            
            # 更新缓存
            updated_agent = AgentInfo(**update_data)
            self._agents_cache[agent_id] = updated_agent
            
            logger.info(f"智能体更新成功: {agent_id}")
            return updated_agent
            
        except Exception as e:
            logger.error(f"更新智能体失败: {e}")
            return None
    
    def delete_agent(self, agent_id: str) -> bool:
        """删除智能体"""
        try:
            logger.info(f"开始删除智能体: {agent_id}")
            
            # 从数据库删除
            success = self.storage.delete_agent(agent_id)
            
            if success:
                # 从缓存删除
                self._agents_cache.pop(agent_id, None)
                logger.info(f"智能体删除成功: {agent_id}")
            else:
                logger.warning(f"智能体删除失败: {agent_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"删除智能体失败: {e}")
            return False
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """获取可用智能体列表"""
        try:
            agents = self.list_agents(size=100)  # 获取所有智能体
            
            # 转换为简单格式
            available_agents = []
            for agent in agents:
                if agent.status == "active":
                    available_agents.append({
                        "agent_id": agent.agent_id,
                        "name": agent.name,
                        "agent_type": agent.agent_type,
                        "routing_tags": agent.routing_tags,
                        "bind_kb_ids": agent.bind_kb_ids,
                        "tools": agent.tools
                    })
            
            return available_agents
            
        except Exception as e:
            logger.error(f"获取可用智能体失败: {e}")
            return []
    
    def test_agent(self, agent_id: str, test_query: str, test_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """测试智能体"""
        try:
            logger.info(f"开始测试智能体: {agent_id}")
            
            # 获取智能体信息
            agent_info = self.get_agent(agent_id)
            if not agent_info:
                return {
                    "success": False,
                    "error": "智能体不存在"
                }
            
            # 执行测试
            test_result = {
                "agent_id": agent_id,
                "agent_name": agent_info.name,
                "test_query": test_query,
                "test_context": test_context,
                "system_prompt": agent_info.system_prompt,
                "tools": agent_info.tools,
                "bind_kb_ids": agent_info.bind_kb_ids,
                "routing_tags": agent_info.routing_tags,
                "test_time": datetime.now(),
                "success": True
            }
            
            logger.info(f"智能体测试完成: {agent_id}")
            return test_result
            
        except Exception as e:
            logger.error(f"测试智能体失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        try:
            agents = self.list_agents(size=1000)  # 获取所有智能体
            
            stats = {
                "total_agents": len(agents),
                "active_agents": len([a for a in agents if a.status == "active"]),
                "agent_types": {},
                "routing_tags": {}
            }
            
            # 统计智能体类型
            for agent in agents:
                agent_type = agent.agent_type.value
                stats["agent_types"][agent_type] = stats["agent_types"].get(agent_type, 0) + 1
                
                # 统计路由标签
                for tag in agent.routing_tags:
                    stats["routing_tags"][tag] = stats["routing_tags"].get(tag, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"获取智能体统计失败: {e}")
            return {
                "total_agents": 0,
                "active_agents": 0,
                "agent_types": {},
                "routing_tags": {}
            }
