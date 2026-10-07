"""Small arithmetic language for vector signals; never executes Python code."""
import ast
import operator
import re

import numpy as np
import pandas as pd

THESIS_FEATURES = ["system_balance_MWh_known_lag96"] + [
    f"{name}_lag_{lag}" for name in ("temperature", "humidity", "weather_wind_speed", "gas_price")
    for lag in (96, 192, 672)
] + ["gas_price_change_1d_known_lag96", "gas_price_roll_mean_7d_known_lag96", "load_fc"]
VARIABLES = set(THESIS_FEATURES) | {"wind", "solar", "demand", "outages", "forecast", "baseline"}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
             ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate_formula(expression: str, inputs: dict[str, pd.Series], index) -> pd.Series:
    expression = re.sub(r"^\s*signal\s*=\s*", "", expression).strip()
    if not expression or len(expression) > 500:
        raise ValueError("Enter a formula of 1–500 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid formula. Example: signal = demand + outages - wind - solar") from exc
    if sum(1 for _ in ast.walk(tree)) > 120:
        raise ValueError("Formula is too complex; use fewer operations.")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
            if not np.isfinite(value) or abs(value) > 1e9:
                raise ValueError("Formula constants must be finite and within ±1 billion.")
            return value
        if isinstance(node, ast.Name):
            if node.id not in VARIABLES:
                raise ValueError(f"Unknown variable '{node.id}'. Use: {', '.join(sorted(VARIABLES))}.")
            if node.id not in inputs:
                raise ValueError(f"Missing '{node.id}' data. Upload a fundamentals CSV containing this column.")
            return inputs[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Div) and np.any(np.asarray(right) == 0):
                raise ValueError("Division by zero in formula.")
            return OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            counts = {"abs": (1, 1), "sign": (1, 1), "min": (2, 8), "max": (2, 8), "avg": (2, 8), "clamp": (3, 3)}
            if name not in counts or not counts[name][0] <= len(node.args) <= counts[name][1]:
                raise ValueError("Use abs(x), sign(x), min/max/avg(x,y,...), or clamp(x,low,high).")
            args = [visit(arg) for arg in node.args]
            if name == "abs":
                return np.abs(args[0])
            if name == "sign":
                return np.sign(args[0])
            if name == "avg":
                return sum(args) / len(args)
            if name == "clamp":
                if np.any(np.asarray(args[1]) > np.asarray(args[2])):
                    raise ValueError("clamp lower bound must not exceed upper bound.")
                return np.minimum(np.maximum(args[0], args[1]), args[2])
            value = args[0]
            for arg in args[1:]:
                value = (np.minimum if name == "min" else np.maximum)(value, arg)
            return value
        raise ValueError("Only numbers, supported variables, + - * /, parentheses and listed functions are allowed.")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = pd.Series(visit(tree.body), index=index, dtype=float)
    if np.isinf(result).any():
        raise ValueError("Formula produced an infinite value. Reduce the scale of the expression.")
    return result
