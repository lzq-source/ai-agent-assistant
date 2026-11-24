from datetime import datetime
import ast
import operator
from pathlib import Path

from langchain_core.tools import tool

# my_tools.py 所在的目录,也就是项目根目录
_BASE_DIR = Path(__file__).resolve().parent

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def _eval_node(node):
    """只计算数字,括号和基础运算符组成的表达式"""
    if isinstance(node,ast.Constant):
        if isinstance(node.value,bool) or not isinstance(node.value,(int,float)):
            raise ValueError("只支持数字常量")
        return node.value

    if isinstance(node,ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if isinstance(node.op,ast.Pow):
            if not isinstance(right,int) or abs(right) > 100:
                raise ValueError("指数必须是 -100 到 100 之间的整数")
        return _BIN_OPS[type(node.op)](left,right)

    if isinstance(node,ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("只支持数字,括号和 + - * / % **")

@tool
def get_current_time() ->str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def read_file(path:str) ->str:
    """读取项目目录内的文本文件内容
    
    Args:
        path: 相对项目根目录的文件路径，例如 docs/note.txt
    """
    target = (_BASE_DIR / path).resolve()

    try:
        target.relative_to(_BASE_DIR)
    except ValueError:
        return "拒绝访问：只能读取项目目录内的文件"

    if not target.is_file():
        return f"文件不存在：{path}"

    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "读取失败：文件不是 UTF-8 文本"

@tool
def calculator(expression: str) ->str:
    """计算数学表达式，支持 + - * / % ** 和括号，例如（12+8）*3

    Args:
        expression: 只包含数字、括号和数字运算符的表达式
    """

    try:
        tree = ast.parse(expression,mode="eval")
        return str(_eval_node(tree.body))
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError) as exc:
        return f"计算失败：{exc}"