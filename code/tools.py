"""STORM 研究助手工具定义

本模块定义研究过程中使用的各种工具。
"""

from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import ArxivRetriever
from langchain_core.runnables import RunnableConfig
from code.configuration import Configuration


class SearchTools:
    """管理研究搜索工具的类"""

    def __init__(self, config: Optional[RunnableConfig] = None):
        """初始化搜索工具

        参数:
            config: 运行时配置
        """
        self.configuration = Configuration.from_runnable_config(config)

        # 初始化 Tavily 搜索工具
        self.tavily_search = TavilySearchResults(
            max_results=self.configuration.tavily_max_results
        )

        # 初始化 ArXiv 搜索工具
        self.arxiv_retriever = ArxivRetriever(
            load_max_docs=self.configuration.arxiv_max_docs,
            load_all_available_meta=True,
            get_full_documents=True,
        )

    async def search_web(self, query: str) -> str:
        """在网络上搜索信息

        参数:
            query: 搜索查询

        返回:
            格式化的搜索结果
        """
        try:
            # 使用 Tavily API 进行网络搜索
            search_results = await self.tavily_search.ainvoke(query)

            # 将结果格式化为文档
            formatted_results = []
            for doc in search_results:
                formatted_doc = (
                    f'<文档 href="{doc["url"]}"/>\n'
                    f'{doc["content"]}\n'
                    f'</文档>'
                )
                formatted_results.append(formatted_doc)

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            return f"<错误>网络搜索时发生错误：{str(e)}</错误>"

    async def search_arxiv(self, query: str) -> str:
        """在 ArXiv 上搜索学术论文

        参数:
            query: 搜索查询

        返回:
            格式化的搜索结果
        """
        try:
            # 在 ArXiv 上搜索论文
            arxiv_results = await self.arxiv_retriever.ainvoke(query)

            # 将结果格式化为文档
            formatted_results = []
            for doc in arxiv_results:
                metadata = doc.metadata
                formatted_doc = (
                    f'<文档 source="{metadata["entry_id"]}" '
                    f'date="{metadata.get("Published", "")}" '
                    f'authors="{metadata.get("Authors", "")}"/>\n'
                    f'<标题>\n{metadata["Title"]}\n</标题>\n\n'
                    f'<摘要>\n{metadata["Summary"]}\n</摘要>\n\n'
                    f'<内容>\n{doc.page_content}\n</内容>\n'
                    f'</文档>'
                )
                formatted_results.append(formatted_doc)

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            return f"<错误>ArXiv 搜索时发生错误：{str(e)}</错误>"


# 工具实例创建函数
def get_search_tools(config: Optional[RunnableConfig] = None) -> SearchTools:
    """根据配置返回搜索工具实例

    参数:
        config: 运行时配置

    返回:
        SearchTools 实例
    """
    return SearchTools(config)
