import agent_lexer
import agent_semantic_analyzer
from agent_lexer import Token
from typing import Any, Dict, Tuple, List

# LL(1) Stack Parser class 

class ParseError(Exception):
    pass


class AgentParser:
    EPS = "ε"

    def __init__(self) -> None:
        #parse table is of the form (NonTerminal, lookahead_terminal) -> production (list of symbols)
        #productions are lists of symbols using "ε" to mean empty.
        self.table: Dict[Tuple[str, str], List[str]] = {}

        #building LL(1) table
        self._build_table()

        self.nonterminals = {
            "program", "agent_defs", "agent_defs'", "agent_def", 
            "agent_block", "capabilities", "cap_stmt", "task_defs", 
            "task_def", "action_block", "action_def", "param_list", 
            "param", "param_tail", "input_list", "input_tail", "type", 
            "system_def", "system_block", "block", "stmt", "for_loop", 
            "if_stmt", "condition", "comparison_op", "var_decl", "run_stmt", 
            "assignment", "expression", "expression'", "term", "term'", 
            "factor", "list_content",  "list_items", "list_tail"}
        
        self.terminals = {
            "AGENT", "SYSTEM", "TOOL", "TASK", "ACTION", "RUN", "FOR", "IN", 
            "IF", "STRING_DECL", "INT_DECL", "LIST_DECL", "BOOL_DECL", "TRUE",
            "FALSE", "ID", "NUM", "STRING_LIT", "LBRACE", "RBRACE", "LPAREN", 
            "RPAREN", "LBRACKET", "RBRACKET", "COMMA", "DOT", "COLON", "PLUS", 
            "MULT","LE", "GE", "EQ", "NEQ", "ARROW", "ASSIGN", "LT", "GT","$"}

    def _add(self, A: str, a: str, rhs: List[str]) -> None:
        key = (A, a)
        
        if key in self.table and self.table[key] != rhs:
            raise ValueError(f"Conflict in parse table at {key}: "
                             f"{self.table[key]} vs {rhs}")
            
        self.table[key] = rhs

    def _build_table(self) -> None:
        # program → agent_defs system_def
        self._add("program", "AGENT", ["agent_defs", "system_def"])

        # agent_defs → agent_def agent_defs'
        self._add("agent_defs", "AGENT", ["agent_def", "agent_defs'"])

        # agent_defs' → agent_def agent_defs' | ε
        self._add("agent_defs'", "AGENT", ["agent_def", "agent_defs'"])
        self._add("agent_defs'", "SYSTEM", [self.EPS])
        # agent_def → agent ID {agent_block}
        self._add("agent_def", "AGENT", ["AGENT", "ID", "LBRACE", "agent_block", "RBRACE"])

        # agent_block → capabilities task_defs
        self._add("agent_block", "TOOL", ["capabilities", "task_defs"])
        self._add("agent_block", "TASK", ["capabilities", "task_defs"])
        self._add("agent_block", "RBRACE", [self.EPS])  # empty agent block 

        # capabilities → cap_stmt capabilities | ε
        self._add("capabilities", "TOOL", ["cap_stmt", "capabilities"])
        self._add("capabilities", "TASK", [self.EPS])
        self._add("capabilities", "RBRACE", [self.EPS])

        # cap_stmt → tool ID
        self._add("cap_stmt", "TOOL", ["TOOL", "ID"])
        
        # task_defs → task_def task_defs | ε
        self._add("task_defs", "TASK", ["task_def", "task_defs"])
        self._add("task_defs", "RBRACE", [self.EPS])

        # task_def → task ID(param_list) -> type ID {action_block}
        self._add("task_def", "TASK", ["TASK", "ID", "LPAREN", "param_list", "RPAREN", "ARROW", "type", "ID", "LBRACE", "action_block", "RBRACE"])

        # action_block → action_def action_block | ε
        self._add("action_block", "ACTION", ["action_def", "action_block"])
        self._add("action_block", "RBRACE", [self.EPS])

        # action_def → action: ID(input_list)
        self._add("action_def", "ACTION", ["ACTION", "COLON", "ID", "LPAREN", "input_list", "RPAREN"])

        # param_list → param param_tail | ε
        self._add("param_list", "STRING_DECL", ["param", "param_tail"])
        self._add("param_list", "INT_DECL", ["param", "param_tail"])
        self._add("param_list", "LIST_DECL", ["param", "param_tail"])
        self._add("param_list", "BOOL_DECL", ["param", "param_tail"])
        self._add("param_list", "RPAREN", [self.EPS])

        # param → type ID
        self._add("param", "STRING_DECL", ["type", "ID"])
        self._add("param", "INT_DECL", ["type", "ID"])
        self._add("param", "LIST_DECL", ["type", "ID"])
        self._add("param", "BOOL_DECL", ["type", "ID"])
        
        # param_tail → , param param_tail | ε
        self._add("param_tail", "COMMA", ["COMMA", "param", "param_tail"])
        self._add("param_tail", "RPAREN", [self.EPS])

        # input_list → expression input_tail | ε
        self._add("input_list", "ID", ["expression", "input_tail"])
        self._add("input_list", "NUM", ["expression", "input_tail"])
        self._add("input_list", "TRUE", ["expression", "input_tail"])
        self._add("input_list", "FALSE", ["expression", "input_tail"])
        self._add("input_list", "LBRACKET", ["expression", "input_tail"])
        self._add("input_list", "RUN", ["expression", "input_tail"])
        self._add("input_list", "STRING_LIT", ["expression", "input_tail"])
        self._add("input_list", "LPAREN", ["expression", "input_tail"])
        self._add("input_list", "RPAREN", [self.EPS])

        # input_tail → , expression input_tail | ε
        self._add("input_tail", "COMMA", ["COMMA", "expression", "input_tail"])
        self._add("input_tail", "RPAREN", [self.EPS])

        # type → string | int | list | bool
        self._add("type", "STRING_DECL", ["STRING_DECL"])
        self._add("type", "INT_DECL", ["INT_DECL"])
        self._add("type", "LIST_DECL", ["LIST_DECL"])
        self._add("type", "BOOL_DECL", ["BOOL_DECL"])

        # system_def → system {system_block}
        self._add("system_def", "SYSTEM", ["SYSTEM", "LBRACE", "system_block", "RBRACE"])

        # system_block → stmt system_block | ε
        self._add("system_block", "FOR", ["stmt", "system_block"])
        self._add("system_block", "IF", ["stmt", "system_block"])
        self._add("system_block", "ID", ["stmt", "system_block"])
        self._add("system_block", "STRING_DECL", ["stmt", "system_block"])
        self._add("system_block", "INT_DECL", ["stmt", "system_block"])
        self._add("system_block", "LIST_DECL", ["stmt", "system_block"])
        self._add("system_block", "BOOL_DECL", ["stmt", "system_block"])
        self._add("system_block", "RBRACE", [self.EPS])

        # block → stmt block | ε
        self._add("block", "FOR", ["stmt", "block"])
        self._add("block", "IF", ["stmt", "block"])
        self._add("block", "ID", ["stmt", "block"])
        self._add("block", "STRING_DECL", ["stmt", "block"])
        self._add("block", "INT_DECL", ["stmt", "block"])
        self._add("block", "LIST_DECL", ["stmt", "block"])
        self._add("block", "BOOL_DECL", ["stmt", "block"])
        self._add("block", "RBRACE", [self.EPS])

        # stmt → for_loop | var_decl | assignment | if_stmt
        self._add("stmt", "FOR", ["for_loop"])
        self._add("stmt", "IF", ["if_stmt"])
        self._add("stmt", "ID", ["assignment"])
        self._add("stmt", "STRING_DECL", ["var_decl"])
        self._add("stmt", "INT_DECL", ["var_decl"])
        self._add("stmt", "LIST_DECL", ["var_decl"])
        self._add("stmt", "BOOL_DECL", ["var_decl"])
        
        # for_loop → for ID in ID {block}
        self._add("for_loop", "FOR", ["FOR", "ID", "IN", "ID", "LBRACE", "block", "RBRACE"])

        # if_stmt → if condition {block}
        self._add("if_stmt", "IF", ["IF", "condition", "LBRACE", "block", "RBRACE"])

        # condition → expression comparison_op expression
        self._add("condition", "ID", ["expression", "comparison_op", "expression"])
        self._add("condition", "NUM", ["expression", "comparison_op", "expression"])
        self._add("condition", "TRUE", ["expression", "comparison_op", "expression"])
        self._add("condition", "FALSE", ["expression", "comparison_op", "expression"])
        self._add("condition", "LBRACKET", ["expression", "comparison_op", "expression"])
        self._add("condition", "RUN", ["expression", "comparison_op", "expression"])
        self._add("condition", "STRING_LIT", ["expression", "comparison_op", "expression"])
        self._add("condition", "LPAREN", ["expression", "comparison_op", "expression"])

        # comparison_op → < | > | == | >= | <= | !=
        self._add("comparison_op", "LT", ["LT"])
        self._add("comparison_op", "GT", ["GT"])
        self._add("comparison_op", "EQ", ["EQ"])
        self._add("comparison_op", "GE", ["GE"])
        self._add("comparison_op", "LE", ["LE"])
        self._add("comparison_op", "NEQ", ["NEQ"])

        # var_decl → type ID = expression
        self._add("var_decl", "STRING_DECL", ["type", "ID", "ASSIGN", "expression"])
        self._add("var_decl", "INT_DECL", ["type", "ID", "ASSIGN", "expression"])
        self._add("var_decl", "LIST_DECL", ["type", "ID", "ASSIGN", "expression"])
        self._add("var_decl", "BOOL_DECL", ["type", "ID", "ASSIGN", "expression"])

        # run_stmt → run ID.ID(input_list)
        self._add("run_stmt", "RUN", ["RUN", "ID", "DOT", "ID", "LPAREN", "input_list", "RPAREN"])

        # assignment → ID = expression
        self._add("assignment", "ID", ["ID", "ASSIGN", "expression"])

        # expression → term expression'
        self._add("expression", "ID", ["term", "expression'"])
        self._add("expression", "NUM", ["term", "expression'"])
        self._add("expression", "TRUE", ["term", "expression'"])
        self._add("expression", "FALSE", ["term", "expression'"])
        self._add("expression", "LBRACKET", ["term", "expression'"])
        self._add("expression", "RUN", ["term", "expression'"])
        self._add("expression", "STRING_LIT", ["term", "expression'"])
        self._add("expression", "LPAREN", ["term", "expression'"])

        # expression' → + term expression' | ε
        self._add("expression'", "PLUS", ["PLUS", "term", "expression'"])
        self._add("expression'", "LT", [self.EPS])
        self._add("expression'", "GT", [self.EPS])
        self._add("expression'", "EQ", [self.EPS])
        self._add("expression'", "GE", [self.EPS])
        self._add("expression'", "LE", [self.EPS])
        self._add("expression'", "NEQ", [self.EPS])
        self._add("expression'", "COMMA", [self.EPS])
        self._add("expression'", "RPAREN", [self.EPS])
        self._add("expression'", "RBRACE", [self.EPS])
        self._add("expression'", "FOR", [self.EPS])
        self._add("expression'", "IF", [self.EPS])
        self._add("expression'", "ID", [self.EPS])
        self._add("expression'", "STRING_DECL", [self.EPS])
        self._add("expression'", "INT_DECL", [self.EPS])
        self._add("expression'", "LIST_DECL", [self.EPS])
        self._add("expression'", "BOOL_DECL", [self.EPS])
        self._add("expression'", "LBRACE", [self.EPS])
        self._add("expression'", "RBRACKET", [self.EPS])

        # term → factor term'
        self._add("term", "ID", ["factor", "term'"])
        self._add("term", "NUM", ["factor", "term'"])
        self._add("term", "TRUE", ["factor", "term'"])
        self._add("term", "FALSE", ["factor", "term'"])
        self._add("term", "LBRACKET", ["factor", "term'"])
        self._add("term", "RUN", ["factor", "term'"])
        self._add("term", "STRING_LIT", ["factor", "term'"])
        self._add("term", "LPAREN", ["factor", "term'"])

        # term' → * factor term' | ε
        self._add("term'", "MULT", ["MULT", "factor", "term'"])
        self._add("term'", "PLUS", [self.EPS])
        self._add("term'", "LT", [self.EPS])
        self._add("term'", "GT", [self.EPS])
        self._add("term'", "EQ", [self.EPS])
        self._add("term'", "GE", [self.EPS])
        self._add("term'", "LE", [self.EPS])
        self._add("term'", "NEQ", [self.EPS])
        self._add("term'", "COMMA", [self.EPS])
        self._add("term'", "RPAREN", [self.EPS])
        self._add("term'", "LBRACE", [self.EPS])
        self._add("term'", "FOR", [self.EPS])
        self._add("term'", "IF", [self.EPS])
        self._add("term'", "ID", [self.EPS])
        self._add("term'", "STRING_DECL", [self.EPS])
        self._add("term'", "INT_DECL", [self.EPS])
        self._add("term'", "LIST_DECL", [self.EPS])
        self._add("term'", "BOOL_DECL", [self.EPS])
        self._add("term'", "RBRACE", [self.EPS])
        self._add("term'", "RBRACKET", [self.EPS])


        # factor → ID | NUM | true | false | list_content | run_stmt | STRING_LIT | (expression)
        self._add("factor", "ID", ["ID"])
        self._add("factor", "NUM", ["NUM"])
        self._add("factor", "TRUE", ["TRUE"])
        self._add("factor", "FALSE", ["FALSE"])
        self._add("factor", "LBRACKET", ["list_content"])
        self._add("factor", "RUN", ["run_stmt"])
        self._add("factor", "STRING_LIT", ["STRING_LIT"])
        self._add("factor", "LPAREN", ["LPAREN", "expression", "RPAREN"])

        # list_content → [list_items]
        self._add("list_content", "LBRACKET", ["LBRACKET", "list_items", "RBRACKET"])

        # list_items → expression list_tail | ε
        self._add("list_items", "ID", ["expression", "list_tail"])
        self._add("list_items", "NUM", ["expression", "list_tail"])
        self._add("list_items", "TRUE", ["expression", "list_tail"])
        self._add("list_items", "FALSE", ["expression", "list_tail"])
        self._add("list_items", "LBRACKET", ["expression", "list_tail"])
        self._add("list_items", "RUN", ["expression", "list_tail"])
        self._add("list_items", "STRING_LIT", ["expression", "list_tail"])
        self._add("list_items", "LPAREN", ["expression", "list_tail"])
        self._add("list_items", "RBRACKET", [self.EPS])

        # list_tail → , expression list_tail | ε
        self._add("list_tail", "COMMA", ["COMMA", "expression", "list_tail"])
        self._add("list_tail", "RBRACKET", [self.EPS])

    def parse(self, tokens: List[Token], trace: bool = False) -> None:
        """
        Table-driven LL(1) parse. Implemented using stack.
        Raises ParseError on failure. On success, consumes input and returns None.
        """
        stack: List[str] = ["$", "program"]
        i = 0

        def lookahead() -> str:
            return tokens[i].type

        while stack:
            top = stack.pop()
            la = lookahead()

            if trace:
                remaining = " ".join(t.type for t in tokens[i:])
                print(f"STACK_TOP={top:>4}  LOOKAHEAD={la:>4}  REMAINING={remaining}")

            #terminal or end marker
            if top in self.terminals:
                if top == la:
                    i += 1
                else:
                    tok = tokens[i]
                    raise ParseError(
                        f"Expected {top}, got {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}"
                    )
                continue

            #epsilon
            if top == self.EPS:
                continue

            #nonterminal
            if top in self.nonterminals:
                prod = self.table.get((top, la))
                if prod is None:
                    tok = tokens[i]
                    expected = sorted({a for (A, a) in self.table.keys() if A == top})
                    raise ParseError(
                        f"No rule for {top} with lookahead {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}. "
                        f"Expected one of: {expected}"
                    )

                #push RHS in reverse (skip ε)
                if len(prod) == 1 and prod[0] == self.EPS:
                    continue
                for sym in reversed(prod):
                    stack.append(sym)
                continue

            raise ParseError(f"Unknown grammar symbol on stack: {top}")

        #if stack is empty we should have consumed '$'
        if tokens[i - 1].type != "$":
            tok = tokens[i]
            raise ParseError(f"Extra input starting at {tok.value!r} (token {tok.type}) at position {tok.line}:{tok.column}")

    def build_ast(self, tokens: List[Token]) -> Dict[str, Any]:
        """
        Validate the input with the existing LL(1) parser, then build an AST.
        The returned AST uses plain dictionaries/lists so the semantic analyzer
        can consume it without depending on parser classes.
        """
        toks = list(tokens)
        if not toks or toks[-1].type != "$":
            toks.append(Token("$", "$", -1, -1))
        self.parse(toks, trace=False)
        return _AstBuilder(toks).parse_program()
    
    def print_ast(self, ast):
        import pprint
        pprint.pprint(ast, width=100, sort_dicts=False)



class _AstBuilder:
    TYPE_NAMES = {
        "STRING_DECL": "string",
        "INT_DECL": "int",
        "LIST_DECL": "list",
        "BOOL_DECL": "bool",
    }

    COMPARISON_OPS = {"LT", "GT", "EQ", "GE", "LE", "NEQ"}

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    def current(self) -> Token:
        return self.tokens[self.i]

    def match(self, token_type: str) -> bool:
        if self.current().type == token_type:
            self.i += 1
            return True
        return False

    def expect(self, token_type: str) -> Token:
        tok = self.current()
        if tok.type != token_type:
            raise ParseError(
                f"Expected {token_type}, got {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}"
            )
        self.i += 1
        return tok

    def parse_program(self) -> Dict[str, Any]:
        agents = []
        while self.current().type == "AGENT":
            agents.append(self.parse_agent_def())

        system = self.parse_system_def()
        self.expect("$")
        return {"node": "program", "agents": agents, "system": system}

    def parse_agent_def(self) -> Dict[str, Any]:
        self.expect("AGENT")
        name = self.expect("ID").value
        self.expect("LBRACE")

        tools = []
        while self.current().type == "TOOL":
            self.expect("TOOL")
            tools.append(self.expect("ID").value)

        tasks = []
        while self.current().type == "TASK":
            tasks.append(self.parse_task_def())

        self.expect("RBRACE")
        return {"node": "agent", "name": name, "tools": tools, "tasks": tasks}

    def parse_task_def(self) -> Dict[str, Any]:
        self.expect("TASK")
        name = self.expect("ID").value
        self.expect("LPAREN")
        params = self.parse_param_list()
        self.expect("RPAREN")
        self.expect("ARROW")
        return_type = self.parse_type()
        return_name = self.expect("ID").value
        self.expect("LBRACE")

        actions = []
        while self.current().type == "ACTION":
            actions.append(self.parse_action_def())

        self.expect("RBRACE")
        return {
            "node": "task",
            "name": name,
            "params": params,
            "return_type": return_type,
            "return_name": return_name,
            "actions": actions,
        }

    def parse_param_list(self) -> List[Dict[str, str]]:
        params = []
        if self.current().type == "RPAREN":
            return params

        params.append(self.parse_param())
        while self.match("COMMA"):
            params.append(self.parse_param())
        return params

    def parse_param(self) -> Dict[str, str]:
        type_name = self.parse_type()
        name = self.expect("ID").value
        return {"type": type_name, "name": name}

    def parse_type(self) -> str:
        tok = self.current()
        if tok.type not in self.TYPE_NAMES:
            raise ParseError(
                f"Expected type, got {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}"
            )
        self.i += 1
        return self.TYPE_NAMES[tok.type]

    def parse_action_def(self) -> Dict[str, Any]:
        self.expect("ACTION")
        self.expect("COLON")
        name = self.expect("ID").value
        self.expect("LPAREN")
        args = self.parse_input_list()
        self.expect("RPAREN")
        return {"node": "action", "name": name, "args": args}

    def parse_input_list(self) -> List[Dict[str, Any]]:
        args = []
        if self.current().type == "RPAREN":
            return args

        args.append(self.parse_expression())
        while self.match("COMMA"):
            args.append(self.parse_expression())
        return args

    def parse_system_def(self) -> Dict[str, Any]:
        self.expect("SYSTEM")
        self.expect("LBRACE")
        statements = self.parse_block()
        self.expect("RBRACE")
        return {"node": "system", "statements": statements}

    def parse_block(self) -> List[Dict[str, Any]]:
        statements = []
        while self.current().type != "RBRACE":
            statements.append(self.parse_stmt())
        return statements

    def parse_stmt(self) -> Dict[str, Any]:
        tok_type = self.current().type
        if tok_type == "FOR":
            return self.parse_for_loop()
        if tok_type == "IF":
            return self.parse_if_stmt()
        if tok_type in self.TYPE_NAMES:
            return self.parse_var_decl()
        if tok_type == "ID":
            return self.parse_assignment()

        tok = self.current()
        raise ParseError(
            f"Expected statement, got {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}"
        )

    def parse_for_loop(self) -> Dict[str, Any]:
        self.expect("FOR")
        variable = self.expect("ID").value
        self.expect("IN")
        iterable = self.expect("ID").value
        self.expect("LBRACE")
        body = self.parse_block()
        self.expect("RBRACE")
        return {"node": "for", "variable": variable, "iterable": iterable, "body": body}

    def parse_if_stmt(self) -> Dict[str, Any]:
        self.expect("IF")
        condition = self.parse_condition()
        self.expect("LBRACE")
        body = self.parse_block()
        self.expect("RBRACE")
        return {"node": "if", "condition": condition, "body": body}

    def parse_condition(self) -> Dict[str, Any]:
        left = self.parse_expression()
        op = self.expect_comparison_op()
        right = self.parse_expression()
        return {"node": "binary", "op": op, "left": left, "right": right}

    def expect_comparison_op(self) -> str:
        tok = self.current()
        if tok.type not in self.COMPARISON_OPS:
            raise ParseError(
                f"Expected comparison operator, got {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}"
            )
        self.i += 1
        return tok.value

    def parse_var_decl(self) -> Dict[str, Any]:
        type_name = self.parse_type()
        name = self.expect("ID").value
        self.expect("ASSIGN")
        value = self.parse_expression()
        return {"node": "var_decl", "type": type_name, "name": name, "value": value}

    def parse_assignment(self) -> Dict[str, Any]:
        name = self.expect("ID").value
        self.expect("ASSIGN")
        value = self.parse_expression()
        return {"node": "assignment", "name": name, "value": value}

    def parse_expression(self) -> Dict[str, Any]:
        expr = self.parse_term()
        while self.match("PLUS"):
            right = self.parse_term()
            expr = {"node": "binary", "op": "+", "left": expr, "right": right}
        return expr

    def parse_term(self) -> Dict[str, Any]:
        expr = self.parse_factor()
        while self.match("MULT"):
            right = self.parse_factor()
            expr = {"node": "binary", "op": "*", "left": expr, "right": right}
        return expr

    def parse_factor(self) -> Dict[str, Any]:
        tok = self.current()

        if self.match("ID"):
            return {"node": "var", "name": tok.value}
        if self.match("NUM"):
            return {"node": "literal", "type": "int", "value": int(tok.value)}
        if self.match("TRUE"):
            return {"node": "literal", "type": "bool", "value": True}
        if self.match("FALSE"):
            return {"node": "literal", "type": "bool", "value": False}
        if self.match("STRING_LIT"):
            return {"node": "literal", "type": "string", "value": tok.value[1:-1]}
        if tok.type == "LBRACKET":
            return self.parse_list_literal()
        if tok.type == "RUN":
            return self.parse_run_call()
        if self.match("LPAREN"):
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr

        raise ParseError(
            f"Expected expression, got {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}"
        )

    def parse_list_literal(self) -> Dict[str, Any]:
        self.expect("LBRACKET")
        items = []
        if self.current().type != "RBRACKET":
            items.append(self.parse_expression())
            while self.match("COMMA"):
                items.append(self.parse_expression())
        self.expect("RBRACKET")
        return {"node": "list", "items": items}

    def parse_run_call(self) -> Dict[str, Any]:
        self.expect("RUN")
        agent = self.expect("ID").value
        self.expect("DOT")
        task = self.expect("ID").value
        self.expect("LPAREN")
        args = self.parse_input_list()
        self.expect("RPAREN")
        return {"node": "run", "agent": agent, "task": task, "args": args}


def build_ast(tokens: List[Token]) -> Dict[str, Any]:
    return AgentParser().build_ast(tokens)


#testing/demo on two examples: first example is from the slides, the second is original. both are accepted by the parser:

def main() -> None:
    samples = [
        """agent Researcher {
            tool web_search
            tool llm
            task gather(string topic) -> string data {
                action: web_search(topic)
                action: llm("summarize results")
            }
        }
            agent Analyzer {
                tool llm
                task sentiment(string text) -> string result {
                action: llm("detect sentiment")
            }
        }
            system {
                list topics = ["AI","Robotics","Security"]
                int i = 0
                bool negative_found = false
                for t in topics {
                    string data = run Researcher.gather(t)
                    string sentiment = run Analyzer.sentiment(data)
                    if sentiment == "negative" {
                    negative_found = true
                }
                i = i + 1
            }
        }""",
        """agent ArticleWriter {
                tool web_search
                tool llm
                task gather_facts(string topic) -> string data {
                    action: web_search(topic)
                }

                task write(string facts, bool allow) -> string article {
                    action: llm("write an article based on these results", allow) 
                }
            }

        agent CurrentEventFinder {
            tool web_search
            task return_recent_events() -> list events {
                action: web_search("current events")
            }
        }

        system {
            list events = run CurrentEventFinder.return_recent_events()
            int i = 0
            if events!=[] {
                for x in events {
                    if x !="" {
                        i = i + 1
                        string facts = run ArticleWriter.gather_facts(x)
                        string article = run ArticleWriter.write(facts, true)
                    }
                }
            }
        }
    """
    ]

    parser = AgentParser()

    for s in samples:
        print("\nINPUT:", s)
        try:
            toks = agent_lexer.tokenize(s) 
            toks.append(Token("$", "$", -1, -1))
            ast= parser.build_ast(toks)
#            parser.print_ast(ast)
            result = agent_semantic_analyzer.analyze(ast)
            print("Semantic analysis accepted")
            print("Agents:")
            for agent_name, agent_info in result["agents"].items():
                print(" ", agent_name)
                print("   tools:", sorted(agent_info["tools"]))
                print("   tasks:")
                for task_name, task_info in agent_info["tasks"].items():
                    params = ", ".join(
                        f"{param['type']} {param['name']}" for param in task_info["params"]
                    )
                    print(
                        f"    {task_name}({params}) -> "
                        f"{task_info['return_type']} {task_info['return_name']}"
                    )

            print("Variables:")
            for name, type_name in result["variables"].items():
                print(" ", name, ":", type_name)

        except (agent_lexer.LexerError, ParseError) as e:
            print("Input rejected:", e)


if __name__ == "__main__":
    main()
