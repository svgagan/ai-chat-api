# mcp_servers/calculator_server.py
from mcp.server.mcpserver import MCPServer
import ast
import operator

mcp = MCPServer("calculator-service")

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        return ALLOWED_OPERATORS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        return ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {type(node).__name__}")

@mcp.tool()
def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression safely."""
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree.body)
        return {"expression": expression, "result": result, "error": None}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8002)