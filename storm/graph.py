"""STORM 研究助手主图定义

本模块定义协调研究过程的 LangGraph 图。
"""

from typing import List, Literal, cast
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# from langgraph.checkpoint.memory import InMemorySaver  # LangGraph API 自动处理此功能
from langgraph.constants import Send

from storm.state import (
    InterviewState,
    InputState,
    OutputState,
    ResearchGraphState,
    Analyst,
    Perspectives,
    SearchQuery,
)
from storm.prompts import (
    ANALYST_INSTRUCTIONS,
    QUESTION_INSTRUCTIONS,
    ANSWER_INSTRUCTIONS,
    SEARCH_INSTRUCTIONS,
    SECTION_WRITER_INSTRUCTIONS,
    REPORT_WRITER_INSTRUCTIONS,
    INTRO_CONCLUSION_INSTRUCTIONS,
)
from storm.configuration import Configuration
from storm.tools import get_search_tools
from storm.utils import load_chat_model, generate_thread_id


# ====================== 分析师生成节点 ======================


async def create_analysts(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """生成针对研究主题定制的分析师人设

    每位分析师都以独特的视角和专业领域为研究做出贡献。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    topic = state["messages"][-1].content
    max_analysts = state.get("max_analysts", configuration.max_analysts)

    # 配置模型以输出结构化结果
    structured_model = model.with_structured_output(Perspectives)

    # 构建提示词
    system_message = ANALYST_INSTRUCTIONS.format(
        topic=topic,
        human_analyst_feedback="",  # 用户反馈为空
        max_analysts=max_analysts,
    )

    # 生成分析师
    result = await structured_model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="生成分析师集合。"),
        ]
    )

    return {"analysts": result.analysts}


# ====================== 访谈节点 ======================


async def generate_question(state: InterviewState, config: RunnableConfig) -> dict:
    """从分析师向专家生成问题

    根据分析师的人设和之前的对话内容创建有洞察力的问题。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    analyst = state["analyst"]
    messages = state["messages"]

    # 构建问题生成提示词
    system_message = QUESTION_INSTRUCTIONS.format(goals=analyst.persona)

    # 生成问题
    question = await model.ainvoke([SystemMessage(content=system_message)] + messages)

    return {"messages": [question]}


async def search_web(state: InterviewState, config: RunnableConfig) -> dict:
    """在网络上搜索相关信息

    分析对话内容以生成适当的搜索查询，
    并在网络上搜索相关信息。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    search_tools = get_search_tools(config)

    # 生成搜索查询
    structured_model = model.with_structured_output(SearchQuery)
    search_query = await structured_model.ainvoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"]
    )

    # 执行网络搜索
    search_results = await search_tools.search_web(search_query.search_query)

    return {"context": [search_results]}


async def search_arxiv(state: InterviewState, config: RunnableConfig) -> dict:
    """在 ArXiv 上搜索学术论文

    分析对话内容以生成适当的搜索查询，
    并在 ArXiv 上搜索相关论文。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    search_tools = get_search_tools(config)

    # 生成搜索查询
    structured_model = model.with_structured_output(SearchQuery)
    search_query = await structured_model.ainvoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"]
    )

    # 执行 ArXiv 搜索
    search_results = await search_tools.search_arxiv(search_query.search_query)

    return {"context": [search_results]}


async def generate_answer(state: InterviewState, config: RunnableConfig) -> dict:
    """以专家身份生成问题的回答

    基于搜索到的上下文，从专家角度创建详细准确的回答。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # 构建回答生成提示词
    system_message = ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)

    # 生成回答
    answer = await model.ainvoke([SystemMessage(content=system_message)] + messages)

    # 标记为专家回答
    answer.name = "expert"

    return {"messages": [answer]}


async def save_interview(state: InterviewState) -> dict:
    """保存完成的访谈内容

    将对话内容转换为字符串格式并保存，用于传递到主图。
    """
    messages = state["messages"]
    interview_content = get_buffer_string(messages)

    # 返回列表形式，以便与主图状态合并
    return {"interview": [interview_content]}


def route_messages(
        state: InterviewState, name: str = "expert"
) -> Literal["ask_question", "save_interview"]:
    """根据访谈进度确定下一步

    达到最大轮数或访谈完成时保存，
    否则生成更多问题。
    """
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 3)

    # 检查专家回复数量
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # 检查是否达到最大轮数
    if num_responses >= max_num_turns:
        return "save_interview"

    # 检查访谈结束信号
    last_question = messages[-2]
    if "非常感谢你的帮助" in last_question.content:
        return "save_interview"

    return "ask_question"


async def write_section(state: InterviewState, config: RunnableConfig) -> dict:
    """根据访谈内容撰写报告章节

    从分析师的角度组织访谈内容，
    撰写报告的一个章节。综合使用搜索结果和专家回答。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    context = state["context"]
    interview = state.get("interview", [])
    analyst = state["analyst"]

    # 将所有搜索结果合并
    formatted_context = "\n\n".join(context) if context else "无搜索结果"

    # 将访谈内容合并（优先使用）
    formatted_interview = "\n\n".join(interview) if interview else "无访谈内容"

    # 构建章节撰写提示词
    system_message = SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)

    # 撰写章节：优先使用访谈内容，搜索结果作为补充
    section = await model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(
                content=f"""请根据以下信息撰写你的章节：

## 访谈内容：
{formatted_interview}

## 参考资料（搜索结果，必要时可引用）：
{formatted_context}"""
            ),
        ]
    )

    return {"sections": [section.content]}


# ====================== 报告撰写节点 ======================


def initiate_all_interviews(state: ResearchGraphState) -> List[Send]:
    """同时启动所有分析师的访谈

    为每位分析师启动独立的访谈流程。
    """
    topic = state.get("topic", "")

    # 为每位分析师启动访谈
    return [
        Send(
            "conduct_interview",
            {
                "analyst": analyst,
                "messages": [
                    HumanMessage(
                        content=f"所以你说你正在写一篇关于 {topic} 的文章？"
                    )
                ],
                "max_num_turns": state.get("max_num_turns", 3),
            },
        )
        for analyst in state["analysts"]
    ]


async def write_report(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """整合所有章节撰写报告正文

    将各分析师撰写的章节综合成连贯的整体报告。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # 连接所有章节
    formatted_sections = "\n\n".join(sections)

    # 构建报告撰写提示词
    system_message = REPORT_WRITER_INSTRUCTIONS.format(
        topic=topic, context=formatted_sections
    )

    # 撰写报告
    report = await model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="根据这些备忘录撰写报告。"),
        ]
    )

    return {"content": report.content}


async def write_introduction(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """撰写报告引言

    创建一个引人入胜的引言，总结整个研究并吸引读者兴趣。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # 连接所有章节
    formatted_sections = "\n\n".join(sections)

    # 构建引言撰写提示词
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, formatted_str_sections=formatted_sections
    )

    # 撰写引言
    intro = await model.ainvoke(
        [instructions, HumanMessage(content="撰写报告引言")]
    )

    return {"introduction": intro.content}


async def write_conclusion(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """撰写报告结论

    创建一个总结关键发现并建议未来研究方向的结论。
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # 连接所有章节
    formatted_sections = "\n\n".join(sections)

    # 构建结论撰写提示词
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, formatted_str_sections=formatted_sections
    )

    # 撰写结论
    conclusion = await model.ainvoke(
        [instructions, HumanMessage(content="撰写报告结论")]
    )

    return {"conclusion": conclusion.content}


async def finalize_report(state: ResearchGraphState) -> dict:
    """组装最终报告

    将引言、正文和结论组合成完整的报告。
    """
    import re

    content = state["content"]

    # 移除"## 洞察"标题
    if content.startswith("## 洞察"):
        content = content.strip("## 洞察")

    # 分离来源部分（使用正则表达式，更灵活地匹配各种格式）
    sources = None
    # 匹配 "## 来源" 前后可能有任意空白字符的情况
    pattern = r'\n*\s*## 来源\s*\n+'
    if "## 来源" in content:
        match = re.search(pattern, content)
        if match:
            # match.end() 之后是来源内容
            sources = content[match.end():].lstrip()
            # 找到来源标题的起始位置，截取正文部分
            source_header_start = content.find("## 来源")
            content = content[:source_header_start].rstrip()

    # 组装最终报告
    final_report = (
            state["introduction"]
            + "\n\n---\n\n## 核心观点\n\n"
            + content
            + "\n\n---\n\n"
            + state["conclusion"]
    )

    # 添加来源部分
    if sources is not None:
        final_report += "\n\n## 来源\n" + sources

    return {
        "final_report": final_report,
        "messages": [HumanMessage(content=final_report)],
    }


# ====================== 图构建函数 ======================


def build_interview_graph():
    """创建访谈子图

    创建一个管理单个分析师访谈流程的子图。
    """
    builder = StateGraph(InterviewState)

    # 添加节点
    builder.add_node("ask_question", generate_question)
    builder.add_node("search_web", search_web)
    builder.add_node("search_arxiv", search_arxiv)
    builder.add_node("answer_question", generate_answer)
    builder.add_node("save_interview", save_interview)
    builder.add_node("write_section", write_section)

    # 定义边
    builder.add_edge(START, "ask_question")
    builder.add_edge("ask_question", "search_web")
    builder.add_edge("ask_question", "search_arxiv")
    builder.add_edge("search_web", "answer_question")
    builder.add_edge("search_arxiv", "answer_question")
    builder.add_conditional_edges(
        "answer_question", route_messages, ["ask_question", "save_interview"]
    )
    builder.add_edge("save_interview", "write_section")
    builder.add_edge("write_section", END)

    # LangGraph API 自动管理检查点
    interview_graph = builder.compile().with_config(run_name="执行访谈")

    return interview_graph


def build_research_graph():
    """创建主研究图

    创建协调整个研究流程的主图。
    """
    # 创建访谈子图
    interview_graph = build_interview_graph()

    # 主图构建器 - 指定输入/输出模式
    builder = StateGraph(ResearchGraphState, input=InputState, output=OutputState)

    # 添加节点
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    # 定义边
    builder.add_edge(START, "create_analysts")
    builder.add_conditional_edges(
        "create_analysts", initiate_all_interviews, ["conduct_interview"]
    )

    # 报告撰写阶段
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")

    # 生成最终报告
    builder.add_edge(
        ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
    )
    builder.add_edge("finalize_report", END)

    # LangGraph API 自动管理检查点
    graph = builder.compile()

    return graph


# ====================== 主图实例 ======================

# 用于 LangGraph Studio 的图实例
graph = build_research_graph()