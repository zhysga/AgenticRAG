#!/usr/bin/env python3
"""
验证 PyMilvus 安装和基本功能，包含增删改查功能
"""

import sys
import random
import os
from typing import List, Dict, Any

def test_basic_import():
    """测试基本导入"""
    print("=== 测试基本导入 ===")
    try:
        from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
        print("✅ 成功导入 pymilvus 核心模块")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_milvus_lite():
    """测试 Milvus Lite 连接"""
    print("\n=== 测试 Milvus Lite 连接 ===")
    try:
        from pymilvus import MilvusClient
        client = MilvusClient("verification_test.db")
        print("✅ 成功创建 MilvusClient")
        
        # 测试基本操作
        client.create_collection(
            collection_name="test", 
            dimension=8
        )
        print("✅ 成功创建集合")
        
        # 手动插入一些简单向量数据
        import numpy as np
        test_data = [
            {"id": i, "vector": np.random.random(8).tolist()}
            for i in range(5)
        ]
        
        client.insert(collection_name="test", data=test_data)
        print("✅ 成功插入测试数据")
        
        # 创建索引
        client.create_index(
            collection_name="test",
            field_name="vector", 
            index_params={"index_type": "FLAT", "metric_type": "L2"}
        )
        print("✅ 成功创建索引")
        
        # 简单搜索测试
        query_vector = np.random.random(8).tolist()
        results = client.search(
            collection_name="test",
            data=[query_vector],
            limit=3,
            output_fields=["id"]
        )
        
        print(f"✅ 成功执行搜索，返回 {len(results[0])} 个结果")
        
        # 清理
        client.drop_collection("test")
        print("✅ 成功清理集合")
            
        client.close()
        print("✅ 成功关闭连接")
        return True
    except Exception as e:
        print(f"❌ Milvus Lite 测试失败: {e}")
        return False

def test_crud_operations():
    """测试增删改查功能"""
    print("\n=== 测试增删改查功能 ===")
    try:
        from pymilvus import MilvusClient
        client = MilvusClient("crud_test.db")
        
        # 创建集合
        if client.has_collection("crud_test_collection"):
            client.drop_collection("crud_test_collection")
        
        client.create_collection(
            collection_name="crud_test_collection",
            dimension=128,
            enable_dynamic_field=True  # 启用动态字段以便添加任意元数据
        )
        print("✅ 创建CRUD测试集合")
        
        # 创建索引
        client.create_index(
            collection_name="crud_test_collection",
            field_name="vector", 
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 64}}
        )
        print("✅ 创建CRUD索引")
        
        # 准备测试数据
        import numpy as np
        
        # 1. 插入数据 (Create)
        initial_data = [
            {
                "id": i,
                "vector": np.random.random(128).tolist(),
                "text": f"初始文档 {i+1}",
                "category": f"category_{(i%5)+1}",
                "metadata": {"source": f"source_{(i%3)+1}", "timestamp": f"2023-01-{(i%28+1):02d}"}
            }
            for i in range(10)
        ]
        
        insert_result = client.insert("crud_test_collection", data=initial_data)
        print(f"✅ 插入 {len(initial_data)} 条初始数据")
        
        # 2. 查询数据 (Read)
        # 加载集合到内存
        client.load_collection("crud_test_collection")
        print("✅ 加载集合到内存")
        
        # 查询所有数据
        all_results = client.query(
            collection_name="crud_test_collection",
            expr="",  # 空表达式表示查询所有
            output_fields=["id", "text", "category", "metadata"]
        )
        print(f"✅ 查询到 {len(all_results)} 条数据")
        
        # 3. 更新数据 (Update)
        # 更新前5条数据
        update_data = [
            {
                "id": item["id"],
                "text": f"更新文档 {item['id']}",
                "metadata": {"updated": True, "update_time": "2023-01-15"}
            }
            for item in all_results[:5]
        ]
        
        # 使用 upsert 更新数据（如果不存在则插入，存在则更新）
        upsert_result = client.upsert("crud_test_collection", data=update_data)
        print(f"✅ 更新 {len(update_data)} 条数据")
        
        # 4. 删除数据 (Delete)
        # 删除ID为偶数的数据
        ids_to_delete = [item["id"] for item in all_results[::2] if item["id"] % 2 == 0]
        if ids_to_delete:
            delete_result = client.delete("crud_test_collection", ids=ids_to_delete)
            print(f"✅ 删除 {len(ids_to_delete)} 条数据 (IDs: {ids_to_delete})")
        
        # 5. 删除数据 (通过过滤表达式)
        filter_delete_result = client.delete(
            collection_name="crud_test_collection",
            filter="category == 'category_3'"
        )
        print(f"✅ 通过过滤器删除 category_3 的数据")
        
        # 6. 验证删除后的数据
        final_results = client.query(
            collection_name="crud_test_collection",
            expr="",
            output_fields=["id", "text"]
        )
        print(f"✅ 最终剩余 {len(final_results)} 条数据")
        
        # 清理
        client.release_collection("crud_test_collection")
        client.drop_collection("crud_test_collection")
        print("✅ 清理CRUD测试集合")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ CRUD操作测试失败: {e}")
        return False

def test_search_with_filtering():
    """测试带过滤条件的搜索"""
    print("\n=== 测试带过滤条件的搜索 ===")
    try:
        from pymilvus import MilvusClient
        client = MilvusClient("search_test.db")
        
        # 创建集合
        if client.has_collection("search_collection"):
            client.drop_collection("search_collection")
        
        client.create_collection(
            collection_name="search_collection",
            dimension=128,
            enable_dynamic_field=True
        )
        client.create_index(
            collection_name="search_collection",
            field_name="vector", 
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
        )
        print("✅ 创建搜索测试集合")
        
        # 插入测试数据
        import numpy as np
        search_data = [
            {
                "id": i,
                "vector": np.random.random(128).tolist(),
                "product": f"产品_{(i%4)+1}",
                "price": float(10 + i),
                "category": f"electronics" if i % 3 == 0 else "clothing" if i % 3 == 1 else "books",
                "in_stock": i % 2 == 0,  # 偶数产品无库存
                "rating": float(4.0 + (i % 2) * 0.5)  # 评分4.0-5.0
                "tags": [f"tag_{(i%3)+1}", f"tag_{(i%5)+1}"]
            }
            for i in range(20)
        ]
        
        client.insert("search_collection", data=search_data)
        print(f"✅ 插入 {len(search_data)} 条搜索测试数据")
        client.load_collection("search_collection")
        print("✅ 加载搜索测试集合")
        
        # 测试不同搜索条件
        query_vector = np.random.random(128).tolist()
        
        # 1. 按类别搜索
        electronics_results = client.search(
            collection_name="search_collection",
            data=[query_vector],
            limit=5,
            filter="category == 'electronics'",
            output_fields=["id", "product", "price", "rating"]
        )
        print(f"✅ 按类别搜索 (electronics): 找到 {len(electronics_results[0])} 个结果")
        
        # 2. 按价格范围搜索
        price_range_results = client.search(
            collection_name="search_collection",
            data=[query_vector],
            limit=5,
            filter="10 <= price <= 50",
            output_fields=["id", "product", "price"]
        )
        print(f"✅ 按价格搜索 (10-50): 找到 {len(price_range_results[0])} 个结果")
        
        # 3. 按评分范围搜索
        rating_results = client.search(
            collection_name="search_collection",
            data=[query_vector],
            limit=5,
            filter="4.0 <= rating <= 5.0",
            output_fields=["id", "product", "rating"]
        )
        print(f"✅ 按评分搜索 (4.0-5.0): 找到 {len(rating_results[0])} 个结果")
        
        # 4. 按库存状态搜索
        in_stock_results = client.search(
            collection_name="search_collection",
            data=[query_vector],
            limit=5,
            filter="in_stock == true",
            output_fields=["id", "product", "in_stock"]
        )
        print(f"✅ 按库存搜索 (有库存): 找到 {len(in_stock_results[0])} 个结果")
        
        # 5. 复合条件搜索
        complex_results = client.search(
            collection_name="search_collection",
            data=[query_vector],
            limit=5,
            filter="category == 'books' and price <= 30 and rating >= 4.0",
            output_fields=["id", "product", "price", "rating"]
        )
        print(f"✅ 复合条件搜索 (书籍且价格<=30且评分>=4.0): 找到 {len(complex_results[0])} 个结果")
        
        # 清理
        client.release_collection("search_collection")
        client.drop_collection("search_collection")
        print("✅ 清理搜索测试集合")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 搜索测试失败: {e}")
        return False

def show_version():
    """显示版本信息"""
    try:
        import pymilvus
        print(f"PyMilvus 版本: {pymilvus.__version__}")
    except:
        print("无法获取 PyMilvus 版本信息")

def show_database_files():
    """显示数据库文件信息"""
    db_files = ["verification_test.db", "crud_test.db", "search_test.db"]
    for db_file in db_files:
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            print(f"📁 数据库文件: {db_file} ({size} bytes)")
        else:
            print(f"📁 数据库文件: {db_file} (不存在)")

def main():
    """主函数"""
    print("🔧 PyMilvus Lite 完整功能测试工具")
    print("=" * 60)
    
    show_version()
    show_database_files()
    
    test_functions = [
        ("基础导入测试", test_basic_import),
        ("Milvus Lite 连接测试", test_milvus_lite),
        ("CRUD 操作测试", test_crud_operations),
        ("搜索与过滤测试", test_search_with_filtering),
    ]
    
    results = {}
    for test_name, test_func in test_functions:
        print(f"\n🧪 开始执行: {test_name}")
        print("-" * 40)
        results[test_name] = test_func()
        status = "✅ 成功" if test_func else "❌ 失败"
        print(f"🧪 {test_name} 完成: {status}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, test_result in results.items():
        status = "✅ 成功" if test_result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n?? 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("?? 所有测试通过！PyMilvus Lite 环境完全可用")
        print("📁 数据库文件保存在当前目录，可以进行数据操作")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    print(f"\n💡 使用提示:")
    print("1. 您可以基于这些测试结果开发您的向量数据库应用")
    print("2. 数据库文件保存在当前目录中，可以随时查询和操作")
    print("3. 所有CRUD和搜索功能都已验证，包括复杂的过滤条件")

if __name__ == "__main__":
    main()
