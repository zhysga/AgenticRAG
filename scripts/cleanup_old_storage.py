#!/usr/bin/env python3
"""
清理旧的存储目录脚本

将所有存储统一到 storage/ 目录后，删除不必要的旧存储文件夹
"""
import os
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 需要删除的旧存储目录
OLD_STORAGE_DIRS = [
    "logs",                    # 旧的日志目录 -> storage/logs
    "backend/logs",            # 后端旧日志目录 -> storage/logs
    "backend/hf_cache",        # 后端HF缓存 -> storage/hf_cache
    "backend/storage",         # 后端旧存储 -> storage/
    "frontend/logs",           # 前端旧日志目录 -> storage/logs
    "frontend/uploads",        # 前端旧上传目录 -> storage/uploads
    "uploads",                 # 旧上传目录 -> storage/uploads
    "chroma",                  # 旧chroma目录 -> storage/chroma_db
    "hf_cache",                # 根目录HF缓存 -> storage/hf_cache
]

# 统一的新存储目录结构
NEW_STORAGE_STRUCTURE = [
    "storage",
    "storage/chroma_db",       # 向量数据库
    "storage/faiss_db",        # FAISS索引
    "storage/uploads",         # 上传文件
    "storage/logs",            # 日志文件
    "storage/hf_cache",        # HuggingFace模型缓存
    "storage/knowledge_bases", # 知识库
    "storage/chat_history",    # 聊天历史
    "storage/data",            # 其他数据
]


def create_new_storage():
    """创建新的存储目录结构"""
    print("📁 创建新的存储目录结构...")
    for dir_path in NEW_STORAGE_STRUCTURE:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")


def cleanup_old_storage(dry_run=True):
    """清理旧的存储目录"""
    print(f"\n{'🔍 预览' if dry_run else '🗑️ 删除'}旧的存储目录...")
    
    deleted_count = 0
    for dir_path in OLD_STORAGE_DIRS:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            if dry_run:
                # 计算目录大小
                size = sum(f.stat().st_size for f in full_path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"   📂 {dir_path} ({size_mb:.2f} MB)")
            else:
                try:
                    shutil.rmtree(full_path)
                    print(f"   ✅ 已删除: {dir_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"   ❌ 删除失败: {dir_path} - {e}")
    
    if not dry_run:
        print(f"\n✅ 共删除 {deleted_count} 个旧目录")


def show_storage_summary():
    """显示存储目录摘要"""
    print("\n📊 存储目录摘要:")
    storage_path = PROJECT_ROOT / "storage"
    
    if storage_path.exists():
        for item in sorted(storage_path.iterdir()):
            if item.is_dir():
                # 计算目录内容
                file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"   📁 {item.name}: {file_count} 文件, {size_mb:.2f} MB")


def main():
    print("=" * 60)
    print("🧹 存储目录清理工具")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    
    # 创建新的存储结构
    create_new_storage()
    
    # 预览要删除的目录
    cleanup_old_storage(dry_run=True)
    
    # 显示存储摘要
    show_storage_summary()
    
    # 确认删除
    print("\n" + "=" * 60)
    confirm = input("是否删除上述旧目录? (输入 'yes' 确认): ")
    
    if confirm.lower() == 'yes':
        cleanup_old_storage(dry_run=False)
        print("\n✅ 清理完成!")
    else:
        print("\n⏸️ 已取消删除操作")
    
    # 最终摘要
    show_storage_summary()


if __name__ == "__main__":
    main()
