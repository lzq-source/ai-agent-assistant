from typing import Annotated,Literal,TypedDict
import operator

from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage,HumanMessage,ToolMessage
from langgraph.graph import StateGraph,START,END
from my_tools import get_current_time,read_file,calculator

tools = [get_current_time,read_file,calculator]
tool_by_name = {t.name:t for t in tools}

model = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://127.0.0.1:11434",
)
model_with_tools = model.bind_tools(tools)

# State:messages 使用追加模式，而不是每次都覆盖
class State(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add]

def llm_call(state:State):
    """节点1：调用模型，由模型决定是否使用工具"""
    return {"messages":[model_with_tools.invoke(state["messages"])]}

def tool_node(state:State):
    """节点2：执行模型请求的工具"""
    out = []
    for tc in state["messages"][-1].tool_calls:
        name = tc["name"]
        print(f"[tool_node] 调用工具：{name} | 参数：{tc['args']}")
        
        t = tool_by_name.get(name)
        
        if t is None:
            result = f"工具不存在：{name}"
        else:
            try:
                result = t.invoke(tc["args"])
            except Exception as exc:
                result = f"工具执行失败：{exc}"
        out.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tc["id"],
                
        )
    )
        
    return {"messages":out}

def should_continue(state: State) -> Literal["tool_node","__end__"]:
    """条件边：最后一条消息有工具调用就去工具节点"""
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END

# 组装图
builder = StateGraph(State)
builder.add_node("llm_call",llm_call)
builder.add_node("tool_node",tool_node)

builder.add_edge(START,"llm_call")
builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node",END],
)
builder.add_edge("tool_node","llm_call")

graph = builder.compile()

# 运行
if __name__ == "__main__":
    questions = [
        "现在几点了？",
        "帮我算（12+8）*3",
        "读一下 docs/note.txt 里面写了什么",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print("问题：",question)
        result = graph.invoke({
            "messages":[
                HumanMessage(content=question)
            ]
        })
        print("\n最终答案：",result["messages"][-1].content)
