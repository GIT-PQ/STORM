"""STORM 研究助手状态定义

本模块定义研究过程各阶段使用的状态。
"""

import operator
from dataclasses import dataclass, field
from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Sequence
from typing_extensions import Annotated


# ====================== 数据模型 ======================


class Analyst(BaseModel):
    """定义分析师属性和元数据的类

    每位分析师都有独特的视角和专业领域。
    """

    # 主要所属机构信息
    affiliation: str = Field(description="分析师的主要所属机构")
    # 姓名
    name: str = Field(description="分析师姓名")
    # 角色
    role: str = Field(description="分析师与研究主题相关的角色")
    # 关注点、关切和动机的描述
    description: str = Field(description="分析师的兴趣、关切和动机描述")

    @property
    def persona(self) -> str:
        """以字符串形式返回分析师的人设"""
        return f"姓名：{self.name}\n角色：{self.role}\n所属机构：{self.affiliation}\n描述：{self.description}\n"


class Perspectives(BaseModel):
    """表示分析师集合的类"""

    analysts: List[Analyst] = Field(
        description="包含角色和所属机构的分析师完整列表"
    )


class SearchQuery(BaseModel):
    """搜索查询的数据类"""

    search_query: str = Field(None, description="用于信息检索的搜索查询")


# ====================== 状态定义 ======================


@dataclass
class InputState:
    """图输入模式"""

    # 研究主题
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


@dataclass
class OutputState:
    """图输出模式"""

    # 完成的最终报告
    final_report: str


class GenerateAnalystsState(InputState):
    """分析师生成阶段的状态"""

    # 研究主题
    topic: str
    # 生成的分析师最大数量
    max_analysts: int
    # 用户反馈
    human_analyst_feedback: Optional[str]
    # 生成的分析师列表
    analysts: List[Analyst]


class InterviewState(MessagesState):
    """访谈阶段的状态

    继承自 MessagesState 以自动管理对话历史。
    """

    # 对话轮数
    max_num_turns: int
    # 包含源文档的上下文列表
    context: Annotated[list, operator.add]
    # 当前正在访谈的分析师
    analyst: Analyst
    # 存储访谈内容的字符串（用于输出到主图）
    interview: Annotated[list, operator.add]
    # 已撰写的报告章节列表
    sections: list


class ResearchGraphState(TypedDict):
    """整个研究过程的内部状态"""

    # 研究主题
    topic: str
    # 生成的分析师最大数量
    max_analysts: int
    # 用户对分析师的反馈
    human_analyst_feedback: Optional[str]
    # 生成的分析师列表
    analysts: List[Analyst]
    # 各分析师的访谈内容（综合上下文和对话）
    interviews: Annotated[list, operator.add]
    # 各分析师撰写的章节
    sections: Annotated[list, operator.add]
    # 最终报告的引言
    introduction: str
    # 最终报告的正文内容
    content: str
    # 最终报告的结论
    conclusion: str
    # 完成的最终报告
    final_report: str

    # 研究主题
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )