"""检查多智能体协作日志

运行方式：
    python scripts/check_agent_logs.py
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "qa_collaboration.jsonl"

def main():
    print("=" * 60)
    print("检查多智能体协作日志")
    print("=" * 60)
    
    if not LOG_FILE.exists():
        print(f"❌ 日志文件不存在: {LOG_FILE}")
        print("请先执行查询生成日志")
        return
    
    print(f"📁 日志文件: {LOG_FILE}\n")
    
    agent_names = set()
    total_lines = 0
    error_lines = 0
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            try:
                log_entry = json.loads(line)
                if log_entry.get("log_type") == "agent_execution":
                    agent_name = log_entry.get("agent_name")
                    agent_names.add(agent_name)
                    print(f"✅ 第{line_num}行: {agent_name} 智能体执行")
            except json.JSONDecodeError:
                error_lines += 1
                print(f"⚠️  第{line_num}行: JSON解析失败（可能被截断）")
    
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总日志行数: {total_lines}")
    print(f"解析错误行数: {error_lines}")
    print(f"检测到的智能体: {sorted(agent_names) if agent_names else '无'}")
    print(f"智能体数量: {len(agent_names)}")
    
    if len(agent_names) >= 3:
        print("\n🎉 多智能体协作正常！检测到至少3个智能体")
    elif len(agent_names) == 1:
        print("\n⚠️  仅检测到1个智能体，多智能体协作未生效")
        print("建议：重启后端服务并重新提交查询")
    elif len(agent_names) > 1:
        print(f"\n⚠️  检测到{len(agent_names)}个智能体，少于预期的3个")
    else:
        print("\n❌ 未检测到智能体执行记录")

if __name__ == "__main__":
    main()
