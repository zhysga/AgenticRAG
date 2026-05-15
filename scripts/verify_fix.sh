#!/bin/bash
# RAG检索修复验证脚本

echo ""
echo "========================================"
echo "RAG检索修复验证"
echo "========================================"
echo ""

# 测试curl命令
echo "测试查询: 13号，周一，我在干嘛"
echo ""

curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "13号，周一，我在干嘛",
    "filters": {
      "custom_filters": {
        "kb_ids": ["a86ea986-70f5-43b6-b3b9-fa58838973fa"]
      }
    },
    "top_k": 3,
    "rerank": true,
    "stream": false
  }' | python -m json.tool

echo ""
echo "========================================"
echo "如果看到citations字段且不为空，说明修复成功！"
echo "========================================"
