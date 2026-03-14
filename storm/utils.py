"""STORM 研究助手工具函数

本模块提供项目中通用的工具函数。
"""

import os
from typing import Union, Optional
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def load_chat_model(model_string: str) -> BaseChatModel:
    """解析模型字符串并加载对应的聊天模型

    参数:
        model_string: 格式为"提供商/模型名称"的字符串

    返回:
        初始化后的聊天模型

    异常:
        ValueError: 如果提供商不受支持
    """
    # 分离提供商和模型名称
    try:
        provider, model_name = model_string.split("/", 1)
    except ValueError:
        raise ValueError(
            f"模型字符串必须是'提供商/模型名称'格式。输入：{model_string}"
        )

    # 根据提供商初始化模型
    if provider == "openai":
        return ChatOpenAI(model=model_name)
    elif provider == "anthropic":
        return ChatAnthropic(model=model_name)
    elif provider == "azure":
        # Azure OpenAI 配置
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")

        if not azure_endpoint or not azure_api_key:
            raise ValueError(
                "使用 Azure OpenAI 时，必须设置 AZURE_OPENAI_ENDPOINT 和 "
                "AZURE_OPENAI_API_KEY 环境变量。"
            )

        return AzureChatOpenAI(
            deployment_name=model_name,
            api_version="2024-12-01-preview",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            temperature=0.1
        )
    elif provider == "bailian":
        # 阿里云百炼配置（OpenAI 兼容接口）
        bailian_api_key = os.environ.get("BAILIAN_API_KEY")

        if not bailian_api_key:
            raise ValueError(
                "使用阿里云百炼时，必须设置 BAILIAN_API_KEY 环境变量。"
            )

        return ChatOpenAI(
            model=model_name,
            api_key=bailian_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.1
        )
    else:
        raise ValueError(f"不支持的提供商：{provider}")


def extract_text_from_message(
    message: Union[AIMessage, HumanMessage, SystemMessage, str]
) -> str:
    """从各种消息类型中提取文本
    
    参数:
        message: 要提取文本的消息
        
    返回:
        提取的文本
    """
    if isinstance(message, str):
        return message
    elif isinstance(message, (AIMessage, HumanMessage, SystemMessage)):
        return message.content
    else:
        return str(message)


def format_analyst_description(analyst) -> str:
    """格式化分析师信息用于显示
    
    参数:
        analyst: 分析师对象
        
    返回:
        格式化的分析师描述
    """
    return (
        f"👤 **{analyst.name}**\n"
        f"   - 角色：{analyst.role}\n"
        f"   - 所属机构：{analyst.affiliation}\n"
        f"   - 专业领域：{analyst.description}"
    )


def format_section_header(section_name: str) -> str:
    """以一致的样式格式化章节标题
    
    参数:
        section_name: 章节名称
        
    返回:
        格式化的标题
    """
    return f"\n\n## {section_name}\n\n"


def truncate_text(text: str, max_length: int = 1000) -> str:
    """将文本截断到指定长度并添加省略号
    
    参数:
        text: 要截断的文本
        max_length: 最大长度
        
    返回:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def clean_source_citation(source: str) -> str:
    """清理来源引用
    
    参数:
        source: 原始来源字符串
        
    返回:
        清理后的来源字符串
    """
    # 移除文档标签
    source = source.replace('<Document source="', '')
    source = source.replace('"/>', '')
    source = source.replace('</Document>', '')
    
    # 移除重复空格
    source = ' '.join(source.split())
    
    return source.strip()


def generate_thread_id() -> str:
    """为检查点生成唯一的线程 ID
    
    返回:
        基于 UUID 的线程 ID
    """
    import uuid
    return str(uuid.uuid4())