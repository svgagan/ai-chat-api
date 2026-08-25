# app/tools/calculator_tool.py
import ast
import operator

# Whitelist of allowed operators only — nothing else is even
# representable, so there's no code execution surface at all
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

def calculate(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree.body)
        return {"expression": expression, "result": result, "error": None}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


CALCULATOR_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluates a mathematical expression. Always use this for any arithmetic beyond trivial single-digit sums, since manual calculation is error-prone for larger numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '847 * 293'"
                }
            },
            "required": ["expression"]
        }
    }
}