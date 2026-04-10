import agent_lexer
from agent_lexer import Token
from typing import Dict, Tuple, List

# LL(1) Stack Parser class

class ParseError(Exception):
    pass


class AgentParser:
    """
    Grammar:
    program → agent_defs system_def

    agent_defs → agent_def agent_defs'
    agent_defs' → agent_def agent_defs' | epsilon
    agent_def → agent ID {agent_block}
    agent_block → capabilities task_defs

    capabilities → cap_stmt capabilities | epsilon
    cap_stmt → tool ID

    task_defs → task_def task_defs | epsilon
    task_def → task ID(param_list) -> type ID {action_block}

    action_block → action_def action_block | epsilon
    action_def → action: ID(input_list)

    param_list → param param_tail | epsilon
    param → type ID
    param_tail → , param param_tail | epsilon

    input_list → expression input_tail | epsilon
    input_tail → , expression input_tail | epsilon

    type → string | int | list | bool

    system_def → system {system_block}
    system_block → stmt system_block | epsilon

    block → stmt block | epsilon
    stmt → for_loop | var_decl | assignment | if_stmt
    for_loop → for ID in ID {block}
    if_stmt → if condition {block}
    condition → expression comparison_op expression
    comparison_op → < | > | == | >= | <= | !=
    var_decl → type ID = expression
    run_stmt → run ID.ID(input_list)
    assignment → ID = expression
    expression → term expression'
    expression' → + term expression' | epsilon
    term → factor term'
    term' → * factor term' | epsilon
    factor → ID | NUM | true | false | list_content | run_stmt | STRING_LIT | (expression) 
    list_content → [list_items]
    list_items → expression list_tail | epsilon
    list_tail → , expression list_tail | epsilon


    Terminals:
        "AGENT", "SYSTEM", "TOOL", "TASK", "ACTION", "RUN", "FOR", "IN", 
        "IF", "STRING_DECL", "INT_DECL", "LIST_DECL", "BOOL_DECL", "TRUE",
        "FALSE", "ID", "NUM", "STRING_LIT", "LBRACE", "RBRACE", "LPAREN", 
        "RPAREN", "LBRACKET", "RBRACKET", "COMMA", "DOT", "COLON", "PLUS", 
        "MULT","LE", "GE", "EQ", "NEQ", "ARROW", "ASSIGN", "LT", "GT","$"

    Nonterminals:
        "program", "agent_defs", "agent_defs'", "agent_def", "agent_block",
        "capabilities", "cap_stmt", "task_defs", "task_def", "action_block",
        "action_def", "param_list", "param", "param_tail", "input_list",
        "input_tail", "type", "system_def", "system_block", "block", "stmt",
        "for_loop", "if_stmt", "condition", "comparison_op", "var_decl",
        "run_stmt", "assignment", "expression", "expression'", "term", "term'",
        "factor", "list_content", "list_items", "list_tail"
    """
    EPS = "ε"

    def __init__(self) -> None:
        # Parse table: (NonTerminal, lookahead_terminal) -> production (list of symbols)
        # Productions are lists of symbols, using "ε" to mean empty.
        self.table: Dict[Tuple[str, str], List[str]] = {}

        # Build a correct LL(1) table for the given grammar.
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

            # Terminal or end marker
            if top in self.terminals:
                if top == la:
                    i += 1
                else:
                    tok = tokens[i]
                    raise ParseError(
                        f"Expected {top}, got {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}"
                    )
                continue

            # Epsilon
            if top == self.EPS:
                continue

            # Nonterminal
            if top in self.nonterminals:
                prod = self.table.get((top, la))
                if prod is None:
                    tok = tokens[i]
                    expected = sorted({a for (A, a) in self.table.keys() if A == top})
                    raise ParseError(
                        f"No rule for {top} with lookahead {tok.type} ({tok.value!r}) at position {tok.line}:{tok.column}. "
                        f"Expected one of: {expected}"
                    )

                # push RHS in reverse (skip ε)
                if len(prod) == 1 and prod[0] == self.EPS:
                    continue
                for sym in reversed(prod):
                    stack.append(sym)
                continue

            raise ParseError(f"Unknown grammar symbol on stack: {top}")

        # If stack is empty, we should have consumed '$'
        if tokens[i - 1].type != "$":
            tok = tokens[i]
            raise ParseError(f"Extra input starting at {tok.value!r} (token {tok.type}) at position {tok.line}:{tok.column}")


# Demo / quick test

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
        }"""
    ]

    parser = AgentParser()

    for s in samples:
        print("\nINPUT:", s)
        try:
            toks = agent_lexer.tokenize(s) 
            toks.append(Token("$", "$", -1, -1))
            parser.parse(toks, trace=False)
            agent_lexer.print_tokens(toks)
            print("Input accepted")
        except (agent_lexer.LexerError, ParseError) as e:
            print("Input rejected:", e)


if __name__ == "__main__":
    main()