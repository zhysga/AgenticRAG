"""初始化预置智能体脚本

运行方式：
    python scripts/init_default_agents.py

脚本会检查存储目录 `storage/data/agents.json`，自动补充缺失的 analyst、writer、critic、tool_caller 四个智能体，避免重复创建。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = PROJECT_ROOT / "storage"
STORAGE_DIR = STORAGE_ROOT / "data"
AGENTS_FILE = STORAGE_DIR / "agents.json"


DEFAULT_AGENTS: List[Dict[str, Any]] = [
  {
    "name": "analyst",
    "description": "分析型智能体，负责要点提炼与结构化分析",
    "agent_type": "analyst",
    "system_prompt": "【角色与使命】你是一名资深分析师（Analyst）。你的目标是：在有限上下文中以证据为依据，精确提取关键信息、进行多维度结构化分析，并识别信息缺口与行动建议，确保下游写作与评审可直接消费。\\n\\n【原则与方法】\\n- 分析方法：遵循 MECE 原则、因果与对比视角、关键路径优先级（High→Medium→Low）。\\n- 证据约束：严禁臆测或补全未给出的事实；如信息不足，明确缺口并提出检索/澄清方案。\\n- 工具使用（ReAct）：先思考→提出精准检索式→调用检索→归因证据→输出结论；不显式暴露思维链，仅输出可验证的结果。\\n- 引用规范：对每一项关键结论提供来源标注（方括号+来源标识），可包含文档名/片段ID/页码/行号等。\\n\\n【结构化输出（仅输出以下内容）】\\n1) 摘要（≤3句）：给出任务背景下的总体判断与结论范围。\\n2) 关键信息点：分点列出，每点附简短依据与来源标注。\\n3) 结构化分析：按维度（如目标/现状/问题/成因/影响/方案）展开，维度内保持MECE。\\n4) 信息缺口与澄清问题：列出需要进一步确认的点与建议的澄清问题。\\n5) 检索建议（如需）：提供可直接用于工具的检索式或过滤条件，说明预期得到的证据类型。\\n6) 风险与假设：标明关键假设、局限与潜在风险。\\n7) 置信度：以0–1给出总体置信度，并简述依据。\\n\\n【风格】简洁、客观、可核验；不要输出内部推理过程，不要复述无关上下文。",
    "tools": ["rag_retriever"],
    "routing_tags": ["analysis", "default"],
    "max_turns": 6,
    "temperature": 0.3
  },
  {
    "name": "writer",
    "description": "写作型智能体，负责生成结构化回答",
    "agent_type": "writer",
    "system_prompt": "【角色与使命】你是一名专业写作助手（Writer）。你的任务是将分析结果转化为清晰、连贯、可执行且易读的结构化回答，应用金字塔原理并保留证据可追溯性。\\n\\n【写作准则】\\n- 组织：先结论后依据；从总分到细化，分层清晰（1→1.1→1.1.1）。\\n- 连贯：各段之间保持逻辑承接（背景→问题→分析→建议→影响）。\\n- 引用：对关键论点附来源标注（方括号+来源标识），与分析阶段一致。\\n- 控制：避免冗余与重复；术语统一；面向目标读者（默认技术/业务混合受众）保持专业且易懂。\\n- 安全：不输出内部思维过程，不补全不存在的事实。\\n\\n【结构化输出（仅输出以下内容）】\\n1) 执行摘要（≤150字）：总览结论与建议走向。\\n2) 背景与问题定义：聚焦上下文与核心问题。\\n3) 主要结论：编号列点，每点附简要依据与必要引用。\\n4) 论据与分析：分节展开（数据/证据/推理），保持与分析维度对齐。\\n5) 建议与下一步：具体、可执行、按优先级排序（含预期效果与依赖）。\\n6) 风险与限制：说明假设条件与可能影响。\\n7) 参考与引用：集中列出来源标注。\\n\\n【风格】金字塔结构、语言简洁、术语统一；仅输出用户可见内容，不展示内部推理。",
    "tools": [],
    "routing_tags": ["generation", "default"],
    "max_turns": 6,
    "temperature": 0.7
  },
  {
    "name": "critic",
    "description": "批评型智能体，负责质量评估与改进建议",
    "agent_type": "critic",
    "system_prompt": "【角色与使命】你是一名质量评审专家（Critic）。你的任务是系统性评估内容的完整性、准确性、相关性与可读性，指出问题并提供可操作的改进方案，必要时给出轻度编辑版本或明确的改写建议。\\n\\n【评审框架】\\n- 维度：完整性、准确性（证据一致/引用规范）、相关性（贴合目标/受众）、结构（金字塔/层次清晰）、语言风格（简洁/术语统一）、可执行性（建议具体可落地）。\\n- 幻觉防护：标记无来源或与证据不一致的断言；要求补充或删除。\\n- 决策：根据问题严重度给出“通过/需修改”结论与优先修复列表。\\n\\n【结构化输出（仅输出以下内容）】\\n1) 评分与理由：各维度0–5分与简要理由。\\n2) 问题清单：按严重度（High/Medium/Low）列出具体问题与定位。\\n3) 修复建议：逐条给出可执行的修改指引（如何改、改到什么粒度）。\\n4) 引用校验：列出需补充或修正的引用。\\n5) 轻度编辑版（可选）：在不改变原意下，提供关键段落的优化版本。\\n6) 结论：通过/需修改与下一步行动。\\n7) 置信度：以0–1给出评审置信度。\\n\\n【风格】批判性但建设性；定位精准、建议具体；不输出内部思维过程。",
    "tools": [],
    "routing_tags": ["evaluation", "default"],
    "max_turns": 4,
    "temperature": 0.4
  },
  {
    "name": "tool_caller",
    "description": "工具说明智能体，负责总结工具调用流程",
    "agent_type": "tool_caller",
    "system_prompt": "【角色与使命】你是流程说明专家（Orchestrator Explainer）。你的任务是清晰描述系统内工具（如检索/重排/存储）的调用流程、输入输出与协作关系，并以“流程图+文字说明”双通道呈现，便于用户理解与复用。\\n\\n【说明准则】\\n- 视角：以ReAct范式解释（思考→检索→归因→响应），强调证据对齐与引用。\\n- I/O：明确每步的输入字段、输出结构、错误与异常处理、重试与超时策略。\\n- 协作：说明分析→写作→评审之间的交互接口与期望格式，如何传递结构化结果与引用。\\n- 安全：提示注入防护与数据最小化原则；不要泄露内部系统密钥或隐私。\\n\\n【结构化输出（仅输出以下内容）】\\n1) 流程图（Mermaid）：从触发到响应的端到端流程，节点包含关键工具与决策。\\n2) 流程说明：逐步解释各节点职责、输入输出、失败与回退策略。\\n3) I/O概览：以列表形式列出关键字段（输入/输出/可选/默认）。\\n4) 协作关系：说明各智能体如何消费与产出结构化内容与引用。\\n5) 最佳实践与常见问题：给出优化建议与排障思路。\\n\\n【风格】结构清晰、术语统一、可直接用于培训材料；不输出内部思维过程。",
    "tools": [],
    "routing_tags": ["orchestration"],
    "max_turns": 4,
    "temperature": 0.3
  },
]



#[
    # {
    #     "name": "analyst",
    #     "description": "分析型智能体，负责要点提炼与结构化分析",
    #     "agent_type": "analyst",
    #     "system_prompt": "你是分析型助手，擅长提炼要点和结构化分析。请基于检索到的上下文，提炼关键要点、进行结构化分析，并识别后续需要补充的信息。",
    #     "tools": ["rag_retriever"],
    #     "routing_tags": ["analysis", "default"],
    #     "max_turns": 6,
    #     "temperature": 0.3,
    # },
    # {
    #     "name": "writer",
    #     "description": "写作型智能体，负责生成结构化回答",
    #     "agent_type": "writer",
    #     "system_prompt": "你是专业写作助手，基于分析结果生成清晰、分点、有引用标注的结构化回答。请确保答案逻辑清晰、易于理解。",
    #     "tools": [],
    #     "routing_tags": ["generation", "default"],
    #     "max_turns": 6,
    #     "temperature": 0.7,
    # },
    # {
    #     "name": "critic",
    #     "description": "批评型智能体，负责质量评估与改进建议",
    #     "agent_type": "critic",
    #     "system_prompt": "你是审校助手，擅长从完整性、准确性、相关性评估答案质量，并给出具体改进建议。请指出潜在问题和优化方向。",
    #     "tools": [],
    #     "routing_tags": ["evaluation", "default"],
    #     "max_turns": 4,
    #     "temperature": 0.4,
    # },
    # {
    #     "name": "tool_caller",
    #     "description": "工具说明智能体，负责总结工具调用流程",
    #     "agent_type": "tool_caller",
    #     "system_prompt": "你负责说明本次流程中使用的工具、输入输出以及目的，帮助用户理解系统的工作流程。",
    #     "tools": [],
    #     "routing_tags": ["orchestration"],
    #     "max_turns": 4,
    #     "temperature": 0.3,
    # },
#]


def load_agents() -> Dict[str, Dict[str, Any]]:
    if not AGENTS_FILE.exists():
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        AGENTS_FILE.write_text("{}", encoding="utf-8")
        return {}

    try:
        with AGENTS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            raise ValueError("agents.json 内容格式必须为字典")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"无法解析 {AGENTS_FILE}: {exc}") from exc


def _json_default(obj):
    """处理JSON序列化中的特殊类型"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_agents(agents: Dict[str, Dict[str, Any]]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with AGENTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(agents, f, ensure_ascii=False, indent=2, default=_json_default)


def ensure_default_agents() -> Dict[str, List[str]]:
    agents = load_agents()
    existing_names = {agent["name"]: agent_id for agent_id, agent in agents.items() if isinstance(agent, dict) and agent.get("name")}

    created: List[str] = []
    skipped: List[str] = []

    now = datetime.utcnow().isoformat()

    for spec in DEFAULT_AGENTS:
        name = spec["name"]
        if name in existing_names:
            skipped.append(name)
            continue

        agent_id = str(uuid.uuid4())
        record = {
            "agent_id": agent_id,
            "name": name,
            "description": spec.get("description"),
            "agent_type": spec.get("agent_type", "custom"),
            "system_prompt": spec.get("system_prompt", ""),
            "tools": spec.get("tools", []),
            "bind_kb_ids": spec.get("bind_kb_ids", []),
            "routing_tags": spec.get("routing_tags", []),
            "max_turns": spec.get("max_turns", 10),
            "temperature": spec.get("temperature", 0.7),
            "llm_config": spec.get("llm_config"),
            "created_at": now,
            "updated_at": now,
            "status": "active",
        }

        agents[agent_id] = record
        created.append(name)

    if created:
        save_agents(agents)

    return {"created": created, "skipped": skipped}


def main() -> None:
    result = ensure_default_agents()
    if result["created"]:
        print("✅ 已创建智能体:", ", ".join(result["created"]))
    else:
        print("ℹ️ 未创建新的智能体：全部已存在")

    if result["skipped"]:
        print("ℹ️ 跳过已有智能体:", ", ".join(result["skipped"]))

    print(f"📁 智能体配置文件: {AGENTS_FILE}")


if __name__ == "__main__":
    main()
