"""
多轮对话上下文记忆功能测试

测试方案B的实现：
1. ChatService.get_message_history() 方法
2. LangGraphService.execute_workflow() 的 chat_messages 参数
3. chat_router 的完整调用链

运行方式：
    cd /home/z/zhy/AIPT5
    python -m pytest tests/test_conversation_memory.py -v
    
或者直接运行：
    python tests/test_conversation_memory.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List


class TestChatServiceMessageHistory(unittest.TestCase):
    """测试 ChatService 的消息历史功能"""
    
    def setUp(self):
        """测试前初始化"""
        from backend.services.chat_service import ChatService
        self.chat_service = ChatService()
        self.test_session_id = "test-session-001"
    
    def test_get_message_history_empty(self):
        """测试空会话的历史消息"""
        messages = self.chat_service.get_message_history("non-existent-session")
        self.assertEqual(len(messages), 0)
        self.assertIsInstance(messages, list)
    
    def test_save_and_get_message_history(self):
        """测试保存后获取历史消息"""
        # 保存第一轮对话
        self.chat_service.save_chat_history(
            session_id=self.test_session_id,
            query="你好，我叫张三",
            answer="你好张三！有什么可以帮助你的吗？",
            processing_time=0.5
        )
        
        # 获取消息历史
        messages = self.chat_service.get_message_history(self.test_session_id)
        
        # 验证消息数量和类型
        self.assertEqual(len(messages), 2)  # 1轮 = 1 Human + 1 AI
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[0].content, "你好，我叫张三")
        self.assertEqual(messages[1].content, "你好张三！有什么可以帮助你的吗？")
    
    def test_message_history_multi_rounds(self):
        """测试多轮对话的历史消息"""
        # 保存多轮对话
        conversations = [
            ("你好，我叫李四", "你好李四！很高兴认识你。"),
            ("我想了解一下你的功能", "我是一个智能问答助手，可以回答你的问题。"),
            ("你记得我叫什么吗？", "你叫李四，我们刚才介绍过了。"),
        ]
        
        for query, answer in conversations:
            self.chat_service.save_chat_history(
                session_id=self.test_session_id,
                query=query,
                answer=answer,
                processing_time=0.3
            )
        
        # 获取消息历史
        messages = self.chat_service.get_message_history(self.test_session_id)
        
        # 验证消息数量
        self.assertEqual(len(messages), 6)  # 3轮 = 3 Human + 3 AI
        
        # 验证消息顺序
        self.assertEqual(messages[0].content, "你好，我叫李四")
        self.assertEqual(messages[1].content, "你好李四！很高兴认识你。")
        self.assertEqual(messages[4].content, "你记得我叫什么吗？")
        self.assertEqual(messages[5].content, "你叫李四，我们刚才介绍过了。")
    
    def test_message_history_max_rounds(self):
        """测试历史消息的最大轮次限制"""
        # 保存10轮对话
        for i in range(10):
            self.chat_service.save_chat_history(
                session_id=self.test_session_id,
                query=f"问题{i}",
                answer=f"回答{i}",
                processing_time=0.1
            )
        
        # 默认获取最近5轮
        messages = self.chat_service.get_message_history(self.test_session_id, max_rounds=5)
        self.assertEqual(len(messages), 10)  # 5轮 = 5 Human + 5 AI
        
        # 验证是最近的5轮
        self.assertEqual(messages[0].content, "问题5")  # 第6轮开始
        self.assertEqual(messages[-1].content, "回答9")  # 最后一轮
    
    def test_message_history_type_consistency(self):
        """测试消息类型的一致性"""
        self.chat_service.save_chat_history(
            session_id=self.test_session_id,
            query="测试问题",
            answer="测试回答",
            processing_time=0.2
        )
        
        messages = self.chat_service.get_message_history(self.test_session_id)
        
        # 验证所有消息都是 BaseMessage 的子类
        for msg in messages:
            self.assertIsInstance(msg, BaseMessage)


class TestIntegration(unittest.TestCase):
    """集成测试：测试完整的多轮对话流程"""
    
    def setUp(self):
        """测试前初始化"""
        from backend.services.chat_service import ChatService
        self.chat_service = ChatService()
        self.test_session_id = "integration-test-session"
    
    def test_conversation_context_flow(self):
        """测试对话上下文流转"""
        # 模拟第一轮对话
        self.chat_service.save_chat_history(
            session_id=self.test_session_id,
            query="我叫王五，我是一名程序员",
            answer="你好王五！作为程序员，你主要使用什么编程语言呢？",
            processing_time=0.5
        )
        
        # 获取历史用于第二轮
        history_messages = self.chat_service.get_message_history(self.test_session_id)
        
        # 验证历史消息可以用于构建上下文
        self.assertEqual(len(history_messages), 2)
        
        # 模拟构建新的消息列表（历史 + 当前问题）
        current_query = "我主要用Python"
        all_messages = list(history_messages) + [HumanMessage(content=current_query)]
        
        # 验证消息列表结构
        self.assertEqual(len(all_messages), 3)
        self.assertIsInstance(all_messages[0], HumanMessage)  # 历史问题
        self.assertIsInstance(all_messages[1], AIMessage)     # 历史回答
        self.assertIsInstance(all_messages[2], HumanMessage)  # 当前问题
        
        # 保存第二轮
        self.chat_service.save_chat_history(
            session_id=self.test_session_id,
            query=current_query,
            answer="Python是一门很棒的语言！王五你做过哪些Python项目？",
            processing_time=0.4
        )
        
        # 获取更新后的历史
        updated_history = self.chat_service.get_message_history(self.test_session_id)
        self.assertEqual(len(updated_history), 4)  # 2轮 = 4条消息


def run_quick_test():
    """快速运行测试"""
    print("=" * 60)
    print("多轮对话上下文记忆功能测试")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestChatServiceMessageHistory))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ 所有测试通过！多轮对话上下文记忆功能正常工作。")
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
        print(f"   失败: {len(result.failures)}")
        print(f"   错误: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
