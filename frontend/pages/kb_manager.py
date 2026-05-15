"""
知识库管理Gradio应用
"""
import gradio as gr
import requests
import json
from typing import List, Dict, Any, Optional
import os
import logging
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

# JSON Schema 兼容性补丁
try:
    from gradio_client import utils as gradio_utils
    _orig_json_schema_to_python_type = getattr(gradio_utils, "_json_schema_to_python_type", None)
    if callable(_orig_json_schema_to_python_type):
        def _safe_json_schema_to_python_type(schema, defs):  # type: ignore[override]
            if isinstance(schema, bool):
                return "Any" if schema else "Never"
            return _orig_json_schema_to_python_type(schema, defs)

        gradio_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
        logger.info("已应用Gradio schema补丁")
except ImportError:
    logger.warning("gradio_client未安装，跳过schema补丁")

# 导入配置和服务
from services.api_client import APIClient
from config.frontend_config import FrontendConfig

# 配置
config = FrontendConfig()
api_client = APIClient(config.backend_url)


class KnowledgeBaseManager:
    """知识库管理器"""
    
    def __init__(self):
        self.current_kb_id = None
        self.uploaded_files = []

    def _get_file_path(self, f):
        """将上传的文件对象统一转换为系统可识别的路径字符串"""
        try:
            return os.fspath(f)
        except TypeError:
            p = getattr(f, "name", None)
            if isinstance(p, str):
                return p
            if isinstance(f, dict):
                n = f.get("name")
                if isinstance(n, str):
                    return n
            return None
    
    def create_knowledge_base(self, name, description, labels, scope):
        """创建知识库"""
        try:
            # 解析标签
            label_list = [label.strip() for label in labels.split(",") if label.strip()]
            
            # 构建请求数据
            data = {
                "name": name,
                "description": description,
                "labels": label_list,
                "scope": scope
            }
            
            # 调用API
            response = api_client.create_knowledge_base(data)
            
            if response.get("status") == "success":
                self.current_kb_id = response["data"]["kb_id"]
                return f"✅ 知识库创建成功！\n知识库ID: {self.current_kb_id}\n名称: {name}"
            else:
                return f"❌ 创建失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 创建失败: {str(e)}"
    
    def upload_files(self, kb_id: str, files, labels: str, chunk_size: int, chunk_overlap: int) -> str:
        """上传文件（支持拖拽上传 - gr.File 返回 list[str] 路径列表）"""
        try:
            # 优先使用用户输入的知识库ID，否则使用当前选中的
            target_kb_id = kb_id.strip() if kb_id and kb_id.strip() else self.current_kb_id

            if not target_kb_id:
                return "❌ 请输入知识库ID或先创建知识库"

            if not files:
                return "❌ 请拖拽或选择要上传的文件"

            # gr.File(type="filepath", file_count="multiple") 返回 list[str]
            # 单文件模式下返回单个 str 或 None
            if isinstance(files, str):
                files = [files]
            elif not isinstance(files, (list, tuple)):
                files = list(files) if files else []

            # 过滤掉 None/空值
            files = [f for f in files if f and isinstance(f, str) and f.strip()]

            if not files:
                return "❌ 未检测到有效文件，请重新拖拽或选择"

            # 解析标签
            label_list = [label.strip() for label in labels.split(",") if label.strip()]

            # 构建请求数据
            data = {
                "kb_id": target_kb_id,
                "labels": label_list,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "auto_index": False
            }

            logger.info(f"开始上传文件到知识库 '{target_kb_id}', 文件数量: {len(files)}")

            # 准备文件数据 - gr.File(filepath) 返回的是临时路径字符串列表
            file_data = []
            for i, file_path in enumerate(files):
                try:
                    file_path = file_path.strip()
                    if not file_path or not os.path.exists(file_path):
                        logger.warning(f"文件 {i+1}/{len(files)}: 路径不存在 {file_path}")
                        continue

                    file_name = os.path.basename(file_path)
                    file_ext = os.path.splitext(file_name)[1].lower()
                    file_size = os.path.getsize(file_path)

                    # 纯文本文件：前端读取内容
                    content = None
                    if file_ext not in ('.pdf', '.doc', '.docx', '.xlsx', '.pptx', '.bin', '.zip',
                                        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp'):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            if not content or not content.strip():
                                content = None
                        except Exception:
                            content = None

                    file_data.append({
                        "file_name": file_name,
                        "file_type": file_ext or ".txt",
                        "content": content,
                        "file_size": file_size,
                        "file_path": file_path
                    })

                    mode = "内容" if content else "路径(后端解析)"
                    logger.debug(f"文件 {i+1}/{len(files)}: {file_name} ({file_size} 字节，通过{mode}上传)")

                except Exception as e:
                    logger.error(f"处理文件 {i+1}/{len(files)} 失败: {e}")
                    continue

            if not file_data:
                logger.error("没有有效的文件数据")
                return "❌ 没有成功读取到任何文件"

            logger.info(f"准备上传 {len(file_data)} 个有效文件")

            # 调用API
            response = api_client.upload_files(data, file_data)

            if response.get("status") == "success":
                uploaded_items = response.get("data", []) or []
                uploaded_count = len(uploaded_items)
                logger.info("文件上传成功，返回持久化路径")
                lines = [
                    "✅ 文件上传成功！",
                    f"知识库ID: {target_kb_id}",
                    f"上传文件数: {uploaded_count}",
                    "--- 文件列表 ---"
                ]
                for item in uploaded_items:
                    name = item.get("file_name", "?")
                    path = item.get("file_path") or "(无路径)"
                    size = item.get("file_size", 0)
                    lines.append(f"• {name}  ({size} 字节)\n  → {path}")
                lines.append("\n💡 点击'开始索引'按钮执行索引")
                return "\n".join(lines)
            else:
                error_msg = response.get('message', '未知错误')
                logger.error(f"文件上传失败: {error_msg}")
                return f"❌ 上传失败: {error_msg}"

        except Exception as e:
            logger.error(f"上传文件失败: {e}", exc_info=True)
            return f"❌ 上传失败: {str(e)}"
    
    def _read_file_content(self, file_item):
        """读取文件内容 - 兼容 Gradio 文件上传对象、路径字符串、字典等多种格式"""
        # 1. 优先尝试通过路径读取文件（最可靠的方式）
        file_path = self._resolve_file_path(file_item)
        if file_path and os.path.exists(file_path):
            try:
                # 判断是否为二进制文件
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ('.pdf', '.doc', '.docx', '.xlsx', '.pptx', '.bin', '.zip'):
                    logger.warning(f"文件 {os.path.basename(file_path)} 是二进制格式({ext})，需要后端解析")
                    # 二进制文件不尝试在前端读取，返回标记让后端通过路径读取
                    return None  # 交给后端 file_path 机制处理
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if content and content.strip():
                    logger.debug(f"通过路径读取文件成功: {os.path.basename(file_path)} ({len(content)} 字符)")
                    return content
                else:
                    logger.warning(f"文件路径存在但内容为空: {file_path}")
            except Exception as e:
                logger.warning(f"通过路径读取文件失败: {file_path}, 错误: {e}")

        # 2. 处理字典对象
        if isinstance(file_item, dict):
            content = file_item.get('content') or file_item.get('data')
            if content:
                if isinstance(content, bytes):
                    try:
                        return content.decode('utf-8', errors='ignore')
                    except Exception:
                        return None
                if isinstance(content, str) and content.strip():
                    return content
            return None

        # 3. 处理有 .read() 方法的文件对象（注意：可能只能读一次）
        if hasattr(file_item, 'read'):
            try:
                # 尝试 seek(0) 以防之前已经读过
                if hasattr(file_item, 'seek'):
                    file_item.seek(0)
                content = file_item.read()
                if isinstance(content, bytes):
                    try:
                        return content.decode('utf-8', errors='ignore')
                    except Exception:
                        return None
                if isinstance(content, str) and content.strip():
                    return content
            except Exception as e:
                logger.warning(f"通过 .read() 读取文件失败: {e}")

        return None

    def _resolve_file_path(self, file_item) -> str | None:
        """从各种文件对象格式中解析出文件路径"""
        # Gradio _NamedString 或其他有 name 属性的对象
        if hasattr(file_item, 'name'):
            name = file_item.name
            if isinstance(name, str) and name:
                return name

        # 字符串路径
        if isinstance(file_item, str) and file_item.strip():
            return file_item.strip()

        # 字典格式
        if isinstance(file_item, dict):
            for key in ('path', 'name', 'file_path'):
                val = file_item.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        # os.fspath 兼容（pathlib.Path 等）
        try:
            path = os.fspath(file_item)
            if isinstance(path, str) and path.strip():
                return path.strip()
        except TypeError:
            pass

        return None
    
    def _get_file_name(self, file_item):
        """获取文件名"""
        if isinstance(file_item, str):
            return os.path.basename(file_item)
        
        if isinstance(file_item, dict):
            return file_item.get('name', 'uploaded_file')
        
        if hasattr(file_item, 'name'):
            return os.path.basename(file_item.name)
        
        return 'uploaded_file'
    
    def list_knowledge_bases(self):
        """列出知识库"""
        try:
            response = api_client.list_knowledge_bases()
            
            if response.get("status") == "success":
                kbs = response["data"]
                if not kbs:
                    return "📝 暂无知识库"
                
                kb_list = "📚 知识库列表:\n\n"
                for kb in kbs:
                    kb_list += f"• {kb['name']} (ID: {kb['kb_id']})\n"
                    kb_list += f"  描述: {kb.get('description', '无')}\n"
                    kb_list += f"  标签: {', '.join(kb.get('labels', []))}\n"
                    kb_list += f"  文件数: {kb.get('file_count', 0)}\n"
                    kb_list += f"  分块数: {kb.get('chunk_count', 0)}\n"
                    kb_list += f"  创建时间: {kb.get('created_at', '未知')}\n\n"
                
                return kb_list
            else:
                return f"❌ 获取失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"
    
    def get_knowledge_base_files(self, kb_id):
        """获取知识库文件"""
        try:
            # 验证知识库ID是否为空
            if not kb_id or not kb_id.strip():
                return "❌ 请先输入知识库ID"
            
            response = api_client.get_knowledge_base_files(kb_id)
            
            if response.get("status") == "success":
                files = response["data"]
                if not files:
                    return "📝 该知识库暂无文件"
                
                file_list = f"📁 知识库 {kb_id} 的文件列表:\n\n"
                for file in files:
                    file_list += f"• {file['file_name']}\n"
                    file_list += f"  类型: {file.get('file_type', '未知')}\n"
                    file_list += f"  大小: {file.get('file_size', 0)} 字节\n"
                    if file.get('file_path'):
                        file_list += f"  路径: {file.get('file_path')}\n"
                    file_list += f"  分块数: {file.get('chunk_count', 0)}\n"
                    file_list += f"  状态: {file.get('index_status', '未知')}\n"
                    file_list += f"  上传时间: {file.get('upload_time', '未知')}\n\n"
                
                return file_list
            else:
                return f"❌ 获取失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 获取失败: {str(e)}"
    
    def reindex_knowledge_base(self, kb_id):
        """重新索引知识库"""
        try:
            # 验证知识库ID是否为空
            target_kb_id = (kb_id or "").strip() or self.current_kb_id
            if not target_kb_id or not str(target_kb_id).strip():
                return "❌ 请先输入知识库ID"
            
            resp = api_client.reindex_knowledge_base(str(target_kb_id), file_ids=None)
            status = (resp or {}).get("status")
            msg = (resp or {}).get("message", "")
            if status in ("success", "accepted"):
                base_msg = f"✅ 已受理知识库 {str(target_kb_id)} 的重新索引任务"
                extra = f"\n提示: {msg}" if msg else "\n提示: 任务已在后台执行，可点击'刷新索引状态'查看进度"
                return base_msg + extra
            return f"❌ 重新索引失败: {msg or '未知错误'}"
            
        except Exception as e:
            return f"❌ 重新索引失败: {str(e)}"

    def get_reindex_status(self, kb_id):
        """获取重新索引状态"""
        try:
            # 验证知识库ID是否为空
            target_kb_id = (kb_id or "").strip() or self.current_kb_id
            if not target_kb_id or not str(target_kb_id).strip():
                return "❌ 请先输入知识库ID"
            
            resp = api_client.get_reindex_status(str(target_kb_id))
            if isinstance(resp, dict) and resp.get("status") == "success":
                data = resp.get("data", {}) or {}
                lines = [
                    f"📊 索引状态 - 知识库: {str(target_kb_id)}",
                    f"状态: {data.get('status', 'unknown')}",
                    f"进度: {data.get('progress', 0)}%",
                    f"已处理文件: {data.get('processed_files', 0)}/{data.get('total_files', 0)}",
                    f"总分块: {data.get('total_chunks', 0)}",
                    f"消息: {data.get('message', '') or '无'}"
                ]
                return "\n".join(lines)
            return f"❌ 获取索引状态失败: {resp.get('message', '未知错误') if isinstance(resp, dict) else '未知错误'}"
        except Exception as e:
            return f"❌ 获取索引状态失败: {str(e)}"
    
    def delete_knowledge_base(self, kb_id, confirm):
        """删除知识库"""
        try:
            # 验证知识库ID是否为空
            if not kb_id or not kb_id.strip():
                return "❌ 请先输入知识库ID"
            
            if not confirm:
                return "❌ 请确认删除操作"
            
            response = api_client.delete_knowledge_base(kb_id, confirm=confirm)
            
            if response.get("status") == "success":
                if self.current_kb_id == kb_id:
                    self.current_kb_id = None
                return f"✅ 知识库 {kb_id} 删除成功！"
            else:
                return f"❌ 删除失败: {response.get('message', '未知错误')}"
                
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"


def create_kb_manager_interface():
    """创建知识库管理界面"""
    kb_manager = KnowledgeBaseManager()
    
    with gr.Blocks(title="知识库管理系统", theme=gr.themes.Soft(), css="""
        .main { max-width: 1200px; margin: 0 auto; padding: 10px; }
        
        /* 紧凑表单样式 */
        .compact-form .form-group { margin-bottom: 6px !important; }
        .compact-form label { font-size: 0.9em; font-weight: 500; margin-bottom: 2px; }
        .compact-form input, .compact-form textarea { padding: 6px 10px; }
        
        /* 结果区域样式 */
        .result-box { background: #f8f9fa; border-radius: 8px; }
        .result-box textarea { font-family: 'Consolas', 'Monaco', monospace; font-size: 0.85em; }
        
        /* 自定义拖拽上传区域 - 隐藏默认图标 */
        .upload-area { 
            min-height: 140px !important; 
            border: 2px dashed #c0c4cc !important; 
            border-radius: 12px !important; 
            background: linear-gradient(135deg, #fafbfc 0%, #f0f2f5 100%) !important;
            transition: all 0.3s ease !important;
            position: relative;
        }
        .upload-area:hover { 
            border-color: #4096ff !important; 
            background: linear-gradient(135deg, #f0f7ff 0%, #e6f4ff 100%) !important;
            box-shadow: 0 4px 12px rgba(64, 150, 255, 0.15);
        }
        /* 隐藏Gradio默认上传图标 */
        .upload-area .icon-wrap { display: none !important; }
        .upload-area .upload-text { display: none !important; }
        .upload-area .or-text { display: none !important; }
        
        /* 自定义上传提示文字 */
        .upload-area::before {
            content: "📂 拖拽文件到此处";
            position: absolute;
            top: 35%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.1em;
            color: #606266;
            font-weight: 500;
        }
        .upload-area::after {
            content: "或点击选择文件";
            position: absolute;
            top: 55%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 0.9em;
            color: #909399;
        }
        
        /* 文件列表预览 */
        .file-preview { font-family: monospace; font-size: 0.85em; }
        
        /* 标签页样式优化 */
        .tab-nav { border-bottom: 2px solid #e4e7ed; }
        .tab-item { padding: 10px 20px; font-weight: 500; }
        .tab-item.selected { border-bottom: 2px solid #4096ff; color: #4096ff; }
        
        /* 按钮组样式 */
        .btn-group { gap: 8px; }
        .btn-group button { flex: 1; }
        
        /* 卡片样式 */
        .card { 
            background: #fff; 
            border-radius: 10px; 
            padding: 16px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #ebeef5;
        }
        
        /* 滑块紧凑样式 */
        .slider-compact input[type=range] { height: 4px; }
        .slider-compact label { font-size: 0.85em; }
    """) as interface:
        gr.Markdown("## 📚 知识库管理系统")
        
        with gr.Tabs():
            # ==================== 创建知识库标签页 ====================
            with gr.Tab("📁 创建知识库"):
                with gr.Row(equal_height=True):
                    # 左侧表单区
                    with gr.Column(scale=3, elem_classes=["card"]):
                        gr.Markdown("**📝 基本信息**")
                        with gr.Row():
                            kb_name = gr.Textbox(
                                label="知识库名称", 
                                placeholder="如: 产品文档库",
                                scale=2
                            )
                            kb_scope = gr.Dropdown(
                                choices=["user", "project", "global"],
                                value="user", 
                                label="作用域", 
                                scale=1
                            )
                        kb_description = gr.Textbox(
                            label="知识库描述", 
                            placeholder="简要描述知识库的用途和内容...",
                            lines=2
                        )
                        kb_labels = gr.Textbox(
                            label="标签（逗号分隔）",
                            placeholder="如: 技术文档, API文档, 产品手册",
                        )
                        create_btn = gr.Button("✨ 创建知识库", variant="primary", size="lg")
                    
                    # 右侧结果区
                    with gr.Column(scale=2, elem_classes=["card", "result-box"]):
                        gr.Markdown("**📋 创建结果**")
                        create_result = gr.Textbox(
                            label="", 
                            lines=10, 
                            interactive=False,
                            placeholder="创建结果将显示在这里..."
                        )
                
                create_btn.click(
                    fn=kb_manager.create_knowledge_base,
                    inputs=[kb_name, kb_description, kb_labels, kb_scope],
                    outputs=create_result
                )
            
            # ==================== 上传文件标签页 ====================
            with gr.Tab("⬆️ 上传文件"):
                with gr.Row(equal_height=True):
                    # 左侧表单区
                    with gr.Column(scale=3, elem_classes=["card"]):
                        gr.Markdown("**📌 目标知识库**")
                        kb_id_for_upload = gr.Textbox(
                            label="知识库ID",
                            placeholder="请输入知识库ID（如: kb_abc123）",
                            value=""
                        )
                        
                        gr.Markdown("**📎 文件上传**")
                        file_upload = gr.File(
                            label="",
                            file_count="multiple",
                            type="filepath",
                            elem_classes=["upload-area"],
                        )
                        gr.Markdown(
                            "<p style='color: #909399; font-size: 0.85em; margin-top: -8px;'>"
                            "💡 支持: TXT, MD, PDF, DOCX, JSON, YAML, CSV, HTML, XML 等格式"
                            "</p>",
                            elem_classes=["file-hint"]
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                file_labels = gr.Textbox(
                                    label="文件标签",
                                    placeholder="如: 重要, 参考",
                                    value=""
                                )
                            with gr.Column(scale=1):
                                with gr.Row():
                                    chunk_size = gr.Slider(100, 2000, 512, step=50, label="分块大小")
                                    chunk_overlap = gr.Slider(0, 200, 50, step=10, label="分块重叠")
                        
                        with gr.Row(elem_classes=["btn-group"]):
                            upload_btn = gr.Button("⬆️ 上传文件", variant="primary")
                            index_btn = gr.Button("🔄 开始索引", variant="secondary")
                            status_btn = gr.Button("📊 刷新状态", variant="secondary")
                    
                    # 右侧结果区
                    with gr.Column(scale=2, elem_classes=["card", "result-box"]):
                        gr.Markdown("**📋 操作结果**")
                        upload_result = gr.Textbox(
                            label="", 
                            lines=16, 
                            interactive=False,
                            placeholder="上传和索引结果将显示在这里..."
                        )
                
                upload_btn.click(
                    fn=kb_manager.upload_files,
                    inputs=[kb_id_for_upload, file_upload, file_labels, chunk_size, chunk_overlap],
                    outputs=upload_result
                )
                
                reindex_evt = index_btn.click(
                    fn=kb_manager.reindex_knowledge_base,
                    inputs=kb_id_for_upload,
                    outputs=upload_result
                )
                reindex_evt.then(
                    fn=kb_manager.get_reindex_status,
                    inputs=kb_id_for_upload,
                    outputs=upload_result
                )
                
                status_btn.click(
                    fn=kb_manager.get_reindex_status,
                    inputs=kb_id_for_upload,
                    outputs=upload_result
                )
                
                def update_kb_id_on_create(result_text):
                    if "知识库ID:" in result_text:
                        for line in result_text.split("\n"):
                            if "知识库ID:" in line:
                                return line.split("知识库ID:")[-1].strip()
                    return ""
                
                create_result.change(fn=update_kb_id_on_create, inputs=create_result, outputs=kb_id_for_upload)
            
            # ==================== 管理知识库标签页 ====================
            with gr.Tab("⚙️ 管理知识库"):
                with gr.Row(equal_height=True):
                    # 左侧列表
                    with gr.Column(scale=2, elem_classes=["card"]):
                        gr.Markdown("**📚 知识库列表**")
                        list_btn = gr.Button("🔄 刷新列表", variant="primary", size="sm")
                        kb_list_result = gr.Textbox(
                            label="", 
                            lines=14, 
                            interactive=False,
                            placeholder="知识库列表将显示在这里..."
                        )
                    
                    # 右侧操作
                    with gr.Column(scale=3, elem_classes=["card"]):
                        gr.Markdown("**🔧 知识库操作**")
                        kb_id_input = gr.Textbox(
                            label="知识库ID", 
                            placeholder="请输入要操作的知识库ID"
                        )
                        
                        with gr.Row(elem_classes=["btn-group"]):
                            files_btn = gr.Button("📁 查看文件", variant="secondary")
                            reindex_btn = gr.Button("🔄 重新索引", variant="secondary")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                delete_confirm = gr.Checkbox(label="确认删除")
                            with gr.Column(scale=2):
                                delete_btn = gr.Button("🗑️ 删除知识库", variant="stop")
                        
                        gr.Markdown("**📋 操作结果**")
                        management_result = gr.Textbox(
                            label="", 
                            lines=8, 
                            interactive=False,
                            placeholder="操作结果将显示在这里..."
                        )
                
                list_btn.click(fn=kb_manager.list_knowledge_bases, outputs=kb_list_result)
                
                files_btn.click(fn=kb_manager.get_knowledge_base_files, inputs=kb_id_input, outputs=management_result)
                
                mgmt_reindex_evt = reindex_btn.click(fn=kb_manager.reindex_knowledge_base, inputs=kb_id_input, outputs=management_result)
                mgmt_reindex_evt.then(fn=kb_manager.get_reindex_status, inputs=kb_id_input, outputs=management_result)
                
                delete_btn.click(fn=kb_manager.delete_knowledge_base, inputs=[kb_id_input, delete_confirm], outputs=management_result)
    
    return interface


if __name__ == "__main__":
    interface = create_kb_manager_interface()
    interface.launch(
        server_name=config.gradio_server_name,
        server_port=config.kb_manager_port,
        share=config.gradio_share,
        debug=config.gradio_debug,
        show_api=False
    )