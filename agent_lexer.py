import re #Python’s regular expression engine (the heart of the lexer)
import sys #to read command-line arguments
from collections import namedtuple #creates a lightweight object for tokens

# Token definition
Token = namedtuple("Token", ["type", "value", "line", "column"])

class LexerError(Exception):
    pass

def print_tokens(tokens):
        print("TOKENS:")
        for t in tokens:
            print(f"type={t.type:<4}  value={t.value:<10}  pos=({t.line},{t.column})")


# Token specifications (ORDER MATTERS): implements longest-prefix matching and priority.
# an ordered list of (TOKEN_NAME, REGEX) pairs.
# ORDER MATTERS — earlier rules have priority.
# The lexer tries patterns from top to bottom, and the first matching rule wins.

TOKEN_SPECIFICATION = [
    # Keywords
    ("AGENT", r"agent\b"),
    ("SYSTEM", r"system\b"),
    ("TOOL", r"tool\b"),
    ("TASK", r"task\b"),
    ("ACTION", r"action\b"),
    ("RUN", r"run\b"),
    ("FOR", r"for\b"),
    ("IN", r"in\b"),
    ("IF", r"if\b"),
    ("STRING_DECL", r"string\b"),
    ("INT_DECL", r"int\b"),
    ("LIST_DECL", r"list\b"),
    ("BOOL_DECL", r"bool\b"),
    ("TRUE", r"true\b"),
    ("FALSE", r"false\b"),

    # Literals
    ("NUM", r"(0|[1-9][0-9]*)\b"),
    ("STRING_LIT", r'"([^"]*)"'),

    # Identifiers
    ("ID", r"[a-zA-Z_][a-zA-Z0-9_]*"),


    # Delimiters
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("COMMA", r","),
    ("DOT", r"\."),
    ("COLON", r":"),
    ("PLUS", r"\+"),
    ("MULT", r"\*"),
    ("LE", r"<="),
    ("GE", r">="),
    ("EQ", r"=="),
    ("NEQ", r"!="),
    ("ARROW", r"->"),
    ("ASSIGN", r"="),
    ("LT", r"<"),
    ("GT", r">"),

    # Comments
    ("COMMENT", r"//.*|/\*[\s\S]*?\*/"),

    # Whitespace
    ("WHITESPACE", r"[ \t\r\n]+"),

    # Any other character
    ("MISMATCH", r"."),
]


# Compile master regex
# turns many token regexes into one single regex that can scan the input left-to-right in one pass and 
# tell us which token matched.
# Each (?P<...>...) defines
# a group with a name
# that captures matched text

master_pattern = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPECIFICATION)
)

# Lexer function
def tokenize(code):
    tokens = []
    line_num = 1
    line_start = 0

    #finditer():
    # starts at position 0
    # applies the regex
    # finds a match
    # advances the input pointer
    # repeats until EOF

    for match in master_pattern.finditer(code):
        # lastgroup	What kind of token is this?
        # group()	What text was matched?
        # start()	Where did it occur?
        kind = match.lastgroup
        value = match.group()
        column = match.start() - line_start + 1
        #print (kind, value, column)  # Debug print to trace tokenization

        if kind == "WHITESPACE":
            if "\n" in value:
                line_num += value.count("\n")
                line_start = match.end()
            continue

        if kind == "COMMENT":
            if "\n" in value:
                line_num += value.count("\n")
                line_start = match.end()
            continue

        if kind == "MISMATCH":
            raise SyntaxError(
                f"Unexpected character {value!r} at line {line_num}, column {column}"
            )

        tokens.append(Token(kind, value, line_num, column))

    return tokens

# Main program
def main():
    if len(sys.argv) != 2:
        print("Usage: python lexer.py <source_file>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)

    try:
        tokens = tokenize(code)
    except SyntaxError as e:
        print("Lexical error:", e)
        sys.exit(1)

    for token in tokens:
        print(f"{token.type:<15} {token.value:<15} (line {token.line}, col {token.column})")


if __name__ == "__main__":
    main()
