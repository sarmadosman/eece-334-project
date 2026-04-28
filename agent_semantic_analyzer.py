import agent_lexer
import agent_parser
from agent_lexer import Token
from typing import Any, Dict, Tuple, List


class SemanticError(Exception):
    pass


UNKNOWN="unknown"
LIST_UNKNOWN="list<unknown>"
COMPARISON_OPS={"==", "!=", "<", ">", "<=", ">="}


def analyze(ast: Dict[str, Any]) -> Dict[str, Any]:
    
    agents = build_agent_table(ast)
    variables: Dict[str, str] = {}
    analyze_statements(ast["system"]["statements"], variables, agents)
    
    return {"agents": agents, "variables": variables}


def build_agent_table(ast: Dict[str, Any]) -> Dict[str, Any]:
    
    agents: Dict[str, Any] = {}

    for agent in ast["agents"]:
        
        agent_name = agent["name"]
        
        if agent_name in agents:
            raise SemanticError(f"Duplicate agent {agent_name!r}")

        tools = set()
        
        for tool in agent["tools"]:
            if tool in tools:
                raise SemanticError(f"Duplicate tool {tool!r} in agent {agent_name!r}")
            tools.add(tool)

        tasks = {}
        
        for task in agent["tasks"]:
            task_name = task["name"]
            if task_name in tasks:
                raise SemanticError(f"Duplicate task {task_name!r} in agent {agent_name!r}")

            param_names = set()
            
            for param in task["params"]:
                if param["name"] in param_names:
                    raise SemanticError(
                        f"Duplicate parameter {param['name']!r} in task {agent_name}.{task_name}"
                    )
                param_names.add(param["name"])

            for action in task["actions"]:
                if action["name"] not in tools:
                    raise SemanticError(
                        f"Task {agent_name}.{task_name} uses undeclared tool {action['name']!r}"
                    )

            tasks[task_name] = {
                "params": task["params"],
                "return_type": task["return_type"],
                "return_name": task["return_name"],
                "actions": task["actions"],
            }

        agents[agent_name] = {"tools": tools, "tasks": tasks}

    return agents


def analyze_statements(
    statements: List[Dict[str, Any]], variables: Dict[str, str], agents: Dict[str, Any]
) -> None:
    for stmt in statements:
        analyze_statement(stmt, variables, agents)


def analyze_statement(stmt: Dict[str, Any], variables: Dict[str, str], agents: Dict[str, Any]) -> None:
    node = stmt["node"]

    if node == "var_decl":
        
        name = stmt["name"]
        if name in variables:
            raise SemanticError(f"Variable {name!r} is already declared")

        actual_type = expression_type(stmt["value"], variables, agents)
        require_assignable(stmt["type"], actual_type, f"variable {name!r}")
        
        if stmt["type"] == "list" and is_list_type(actual_type):
            variables[name] = actual_type
        else:
            variables[name] = stmt["type"]
        return

    if node == "assignment":
        
        name = stmt["name"]
        if name not in variables:
            raise SemanticError(f"Assignment to undeclared variable {name!r}")

        actual_type = expression_type(stmt["value"], variables, agents)
        require_assignable(variables[name], actual_type, f"variable {name!r}")
        
        return

    if node == "for":
        iterable = stmt["iterable"]
        if iterable not in variables:
            raise SemanticError(f"For loop uses undeclared iterable {iterable!r}")
        if not is_list_type(variables[iterable]):
            raise SemanticError(f"For loop iterable {iterable!r} must be list, got {variables[iterable]}")

        loop_variables = variables.copy()
        loop_variables[stmt["variable"]] = list_element_type(variables[iterable])
        analyze_statements(stmt["body"], loop_variables, agents)
        return

    if node == "if":
        condition_type = expression_type(stmt["condition"], variables, agents)
        if condition_type != "bool":
            raise SemanticError(f"If condition must be bool, got {condition_type}")

        branch_variables = variables.copy()
        analyze_statements(stmt["body"], branch_variables, agents)
        return

    raise SemanticError(f"Unknown statement node {node!r}")


def expression_type(expr: Dict[str, Any], variables: Dict[str, str], agents: Dict[str, Any]) -> str:
    node = expr["node"]

    if node == "literal":
        return expr["type"]

    if node == "list":
        item_types = [expression_type(item, variables, agents) for item in expr["items"]]
        known_item_types = {item_type for item_type in item_types if item_type != UNKNOWN}

        if not known_item_types:
            return LIST_UNKNOWN
        if len(known_item_types) > 1:
            raise SemanticError(f"List literal has mixed item types: {sorted(known_item_types)}")

        return f"list<{known_item_types.pop()}>"

    if node == "var":
        name = expr["name"]
        if name not in variables:
            raise SemanticError(f"Use of undeclared variable {name!r}")
        return variables[name]

    if node == "run":
        return run_call_type(expr, variables, agents)

    if node == "binary":
        return binary_expression_type(expr, variables, agents)

    raise SemanticError(f"Unknown expression node {node!r}")


def run_call_type(expr: Dict[str, Any], variables: Dict[str, str], agents: Dict[str, Any]) -> str:
    agent_name = expr["agent"]
    task_name = expr["task"]

    if agent_name not in agents:
        raise SemanticError(f"Unknown agent {agent_name!r}")

    tasks = agents[agent_name]["tasks"]
    if task_name not in tasks:
        raise SemanticError(f"Unknown task {agent_name}.{task_name}")

    task = tasks[task_name]
    params = task["params"]
    args = expr["args"]

    if len(args) != len(params):
        raise SemanticError(
            f"{agent_name}.{task_name} expects {len(params)} arguments, got {len(args)}"
        )

    for index, (arg, param) in enumerate(zip(args, params), start=1):
        actual_type = expression_type(arg, variables, agents)
        try:
            require_assignable(param["type"], actual_type, f"argument {index} to {agent_name}.{task_name}")
        except SemanticError as error:
            raise SemanticError(f"{error}; parameter {param['name']!r}") from error

    return task["return_type"]


def binary_expression_type(expr: Dict[str, Any], variables: Dict[str, str], agents: Dict[str, Any]) -> str:
    op = expr["op"]
    left_type = expression_type(expr["left"], variables, agents)
    right_type = expression_type(expr["right"], variables, agents)

    if op == "+":
        if left_type == right_type and (left_type in {"int", "string"} or is_list_type(left_type)):
            return left_type
        if UNKNOWN in {left_type, right_type}:
            return UNKNOWN
        raise SemanticError(f"Operator + does not support {left_type} and {right_type}")

    if op == "*":
        if left_type == "int" and right_type == "int":
            return "int"
        if UNKNOWN in {left_type, right_type}:
            return UNKNOWN
        raise SemanticError(f"Operator * does not support {left_type} and {right_type}")

    if op in COMPARISON_OPS:
        if compatible_types(left_type, right_type) or UNKNOWN in {left_type, right_type}:
            return "bool"
        raise SemanticError(f"Cannot compare {left_type} with {right_type}")

    raise SemanticError(f"Unknown binary operator {op!r}")


def require_assignable(expected_type: str, actual_type: str, context: str) -> None:
    if actual_type == UNKNOWN:
        return
    if compatible_types(expected_type, actual_type):
        return
    if expected_type != actual_type:
        raise SemanticError(f"Cannot assign {actual_type} to {expected_type} {context}")


def compatible_types(expected_type: str, actual_type: str) -> bool:
    if expected_type == actual_type:
        return True
    if is_list_type(expected_type) and is_list_type(actual_type):
        return (
            expected_type == "list"
            or actual_type == "list"
            or expected_type == LIST_UNKNOWN
            or actual_type == LIST_UNKNOWN
        )
    return False


def is_list_type(type_name: str) -> bool:
    return type_name == "list" or type_name.startswith("list<")


def list_element_type(type_name: str) -> str:
    if type_name.startswith("list<") and type_name.endswith(">"):
        return type_name[5:-1]
    return UNKNOWN
