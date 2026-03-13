from graph import graph
import asyncio

from dotenv import load_dotenv

async def main():

    load_dotenv()

    # 定义研究主题
    topic = "人工智能在医疗领域的应用"

    # 配置参数
    config = {
        "configurable": {
            "model": "bailian/qwen-plus",
            "max_analysts": 1,
            "max_interview_turns": 1,
            "tavily_max_results": 3,
            "arxiv_max_docs": 3,
            "parallel_interviews": True,
            "enable_checkpointing": True,
        }
    }

    # 调用图
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": topic}]},
        config=config
    )

    # 输出结果
    print("=" * 80)
    print("研究报告生成完成！")
    print("=" * 80)
    print(result["final_report"])
    return result

if __name__ == "__main__":
    asyncio.run(main())
