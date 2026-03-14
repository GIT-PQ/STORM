"""STORM 研究助手配置管理

本模块管理研究助手系统的运行时配置。
"""

from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig


@dataclass
class Configuration:
    """STORM 研究助手配置类

    定义 LangGraph Studio 中可用的配置选项。
    """

    # 模型设置
    model: str = field(
        default="bailian/qwen-plus",
        metadata={
            "description": "要使用的 LLM 模型（格式：提供商/模型）",
            "examples": [
                "bailian/qwen-plus",
                "bailian/qwen-turbo",
                "bailian/qwen-max",
                "bailian/qwen-long",
                "azure/gpt-4.1",
                "openai/gpt-4.1",
                "openai/gpt-4.1-mini",
                "anthropic/claude-opus-4-20250514",
                "anthropic/claude-3-7-sonnet-latest",
                "anthropic/claude-3-5-haiku-latest",
            ],
        },
    )

    # 研究设置
    max_analysts: int = field(
        default=3,
        metadata={
            "description": "生成的分析师最大数量",
            "range": [1, 10],
        },
    )

    max_interview_turns: int = field(
        default=3,
        metadata={
            "description": "每次访谈的最大对话轮数",
            "range": [1, 10],
        },
    )

    # 搜索设置
    tavily_max_results: int = field(
        default=3,
        metadata={
            "description": "Tavily 搜索结果的最大数量",
            "range": [1, 10],
        },
    )

    arxiv_max_docs: int = field(
        default=3,
        metadata={
            "description": "ArXiv 搜索文档的最大数量",
            "range": [1, 10],
        },
    )

    # 并行处理设置
    parallel_interviews: bool = field(
        default=True, metadata={"description": "是否并行执行访谈"}
    )

    # 检查点设置
    enable_checkpointing: bool = field(
        default=True, metadata={"description": "是否启用状态检查点"}
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 中提取设置创建配置实例

        参数:
            config: 从 LangGraph 传递的运行时配置

        返回:
            应用了设置的配置实例
        """
        configurable = config.get("configurable", {}) if config else {}

        # 创建默认实例以获取默认值
        defaults = cls()

        return cls(
            model=configurable.get("model", defaults.model),
            max_analysts=configurable.get("max_analysts", defaults.max_analysts),
            max_interview_turns=configurable.get(
                "max_interview_turns", defaults.max_interview_turns
            ),
            tavily_max_results=configurable.get(
                "tavily_max_results", defaults.tavily_max_results
            ),
            arxiv_max_docs=configurable.get("arxiv_max_docs", defaults.arxiv_max_docs),
            parallel_interviews=configurable.get(
                "parallel_interviews", defaults.parallel_interviews
            ),
            enable_checkpointing=configurable.get(
                "enable_checkpointing", defaults.enable_checkpointing
            ),
        )