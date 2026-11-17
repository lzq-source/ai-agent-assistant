import ollama

def mul(a:int ,b:int) -> int:
    """计算两个整数的乘积
    
    Args:
        a:第一个整数
        b:第二个整数
    """
    return a * b
    
messages = [
    {"role":"user","content":"帮我算一下 23 乘 45 等于多少"}
]

# 第一次请求：把mul函数作为工具提供给模型
response = ollama.chat(
    model = "qwen2.5:3b",
    messages = messages,
    tools = [mul],
)

print("第一次响应：",response["message"])
messages.append(response["message"])

# 判断模型是否请求调用工具
if response["message"].get("tool_calls"):
    call = response["message"]["tool_calls"][0]

    print("调用了工具，参数：",call["function"]["arguments"])

    # Python 真正执行函数
    result = mul(**call["function"]["arguments"])
    print("Python 计算结果：",result)

    # 把工具执行结果喂回对话
    messages.append({
         "role":"tool",
         "tool_name":call["function"]["name"],
         "content":str(result),
    }) 

    # 第二次请求：模型根据工具组织最终答案
    final = ollama.chat(
        model = "qwen2.5:3b",
        messages = messages,
        tools = [mul],
    )

    print("最终回答：",final["message"]["content"])

else:
    print("模型这次没有调用工具，直接回答：")
    print(response["message"]["content"])