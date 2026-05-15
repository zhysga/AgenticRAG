#!/bin/bash
# PyMilvus Lite 安装和修复脚本

echo "=== PyMilvus Lite 安装和修复脚本 ==="
echo "问题诊断：Milvus 2.6.3 需要 milvus-lite 组件但导入失败"
echo ""

# 检查当前环境
echo "--- 环境检查 ---"
python_version=$(python3 --version 2>&1 | head -n1)
echo "Python版本: $python_version"

pip_version=$(pip --version 2>&1 | head -n1)
echo "Pip版本: $pip_version"

# 检查已安装的包
echo ""
echo "--- 已安装的Milvus相关包 ---"
pip list | grep -E "pymilvus" || echo "未找到pymilvus相关包"

# 解决方案1: 强制重新安装所有必需组件
echo ""
echo "=== 解决方案1: 强制重新安装 ==="
echo "正在卸载现有组件..."
pip uninstall -y pymilvus pymilvus.model pymilvus.bulk_writer 2>/dev/null

echo "正在重新安装基础PyMilvus..."
pip install pymilvus==2.6.3

echo "正在安装模型组件..."
pip install "pymilvus[model]"

echo "正在尝试安装milvus-lite..."
pip install milvus-lite

echo "正在安装bulk_writer..."
pip install "pymilvus[bulk_writer]"

# 解决方案2: 清理pip缓存
echo ""
echo "=== 解决方案2: 清理pip缓存 ==="
pip cache purge

# 解决方案3: 使用conda重新安装
echo ""
echo "=== 解决方案3: 使用conda重新安装 ==="
conda install -c conda-forge pymilvus -y

# 验证安装
echo ""
echo "=== 验证安装 ==="
python3 -c "
try:
    import pymilvus
    print('✅ PyMilvus基础安装成功')
    print('PyMilvus版本:', pymilvus.__version__)
    
    # 检查milvus-lite
    try:
        from pymilvus.client import config
        if hasattr(config, 'MILVUS_LITE_PACKAGE'):
            print('✅ Milvus Lite组件可用')
        else:
            print('❌ Milvus Lite组件配置缺失')
    except ImportError as e:
        print('❌ Milvus Lite组件导入失败:', e)
    
    # 测试本地连接
    try:
        from pymilvus import MilvusClient
        client = MilvusClient('test_fix.db')
        client.close()
        print('✅ 本地连接测试成功')
    except Exception as e:
        print('❌ 本地连接测试失败:', e)
        
except Exception as e:
    print('❌ 验证过程出错:', e)
"

echo ""
echo "=== 手动测试建议 ==="
echo "如果上述自动修复失败，请手动尝试："
echo "1. pip install --force-reinstall pymilvus"
echo "2. pip install 'pymilvus[milvus_lite]'"
echo "3. 或者使用Docker部署Milvus服务器模式"

echo ""
echo "=== 修复完成 ==="
echo "请重新运行您的测试脚本验证结果"
