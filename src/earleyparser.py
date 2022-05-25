"""
Python Earley Parser.

@author: Georgios Sakkas, Earley Parser based on Hardik's implementation
"""

import argparse
import re
# from ast import parse
from pathlib import Path
from collections import defaultdict
from nltk.tree import Tree
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.token import Text, Name, Number, String, Punctuation, Operator, Keyword

class Rule():
    """
    Represents a CFG rule.
    """

    def __init__(self, lhs, rhs):
        # Represents the rule 'lhs -> rhs', where lhs is a non-terminal and
        # rhs is a list of non-terminals and terminals.
        self.lhs, self.rhs = lhs, rhs

    def __contains__(self, sym):
        return sym in self.rhs

    def __eq__(self, other):
        if isinstance(other, Rule):
            return self.lhs == other.lhs and self.rhs == other.rhs

        return False

    def __getitem__(self, i):
        return self.rhs[i]

    def __len__(self):
        return len(self.rhs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.lhs + ' -> ' + ' '.join(self.rhs)


class Grammar():
    """
    Represents a CFG.
    """

    def __init__(self):
        # The rules are represented as a dictionary from L.H.S to R.H.S.
        self.rules = defaultdict(list)

    def add(self, rule):
        """
        Adds the given rule to the grammar.
        """

        self.rules[rule.lhs].append(rule)

    @staticmethod
    def load_grammar(fpath):
        """
        Loads the grammar from file (from the )
        """

        grammar = Grammar()

        with open(fpath) as f:
            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue

                entries = line.split('->')
                lhs = entries[0].strip()
                for rhs in entries[1].split('<|>'):
                    grammar.add(Rule(lhs, rhs.strip().split()))

        return grammar

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = [str(r) for r in self.rules['S']]

        for nt, rule_list in self.rules.items():
            if nt == 'S':
                continue

            s += [str(r) for r in rule_list]

        return '\n'.join(s)

    # Returns the rules for a given Non-terminal.
    def __getitem__(self, nt):
        return self.rules[nt]

    def is_terminal(self, sym):
        """
        Checks is the given symbol is terminal.
        """
        return len(self.rules[sym]) == 0

    def is_tag(self, sym):
        """
        Checks whether the given symbol is a tag, i.e. a non-terminal with rules
        to solely terminals.
        """
        if not self.is_terminal(sym):
            return all(self.is_terminal(s) for r in self.rules[sym] for s in
                r.rhs)

        return False


class State():
    """
    Represents a state in the Earley algorithm.
    """

    GAM = '<GAM>'

    def __init__(self, rule, dot=0, sent_pos=0, chart_pos=0, back_pointers=[]):
        # CFG Rule.
        self.rule = rule
        # Dot position in the rule.
        self.dot = dot
        # Sentence position.
        self.sent_pos = sent_pos
        # Chart index.
        self.chart_pos = chart_pos
        # Pointers to child states (if the given state was generated using
        # Completer).
        self.back_pointers = back_pointers

    def __eq__(self, other):
        if isinstance(other, State):
            return self.rule == other.rule and self.dot == other.dot and \
                self.sent_pos == other.sent_pos

        return False

    def __len__(self):
        return len(self.rule)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        def str_helper(state):
            return ('(' + state.rule.lhs + ' -> ' +
            ' '.join(state.rule.rhs[:state.dot] + ['*'] +
                state.rule.rhs[state.dot:]) +
            (', [%d, %d])' % (state.sent_pos, state.chart_pos)))

        return (str_helper(self) +
            ' {' + ', '.join(str_helper(s) for s in self.back_pointers) + '}')

    def next(self):
        """
        Return next symbol to parse, i.e. the one after the dot
        """

        if self.dot < len(self):
            return self.rule[self.dot]

    def is_complete(self):
        """
        Checks whether the given state is complete.
        """

        return len(self) == self.dot

    @staticmethod
    def init():
        """
        Returns the state used to initialize the chart in the Earley algorithm.
        """

        return State(Rule(State.GAM, ['S']))


class ChartEntry():
    """
    Represents an entry in the chart used by the Earley algorithm.
    """

    def __init__(self, states):
        # List of Earley states.
        self.states = states

    def __iter__(self):
        return iter(self.states)

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n'.join(str(s) for s in self.states)

    def add(self, state):
        """
        Add the given state (if it hasn't already been added).
        """

        if state not in self.states:
            self.states.append(state)


class Chart():
    """
    Represents the chart used in the Earley algorithm.
    """

    def __init__(self, entries):
        # List of chart entries.
        self.entries = entries

    def __getitem__(self, i):
        return self.entries[i]

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n\n'.join([("Chart[%d]:\n" % i) + str(entry) for i, entry in
            enumerate(self.entries)])

    @staticmethod
    def init(l):
        """
        Initializes a chart with l entries (Including the dummy start state).
        """

        return Chart([(ChartEntry([]) if i > 0 else
                ChartEntry([State.init()])) for i in range(l)])


class EarleyParse():
    """
    Represents the Earley-generated parse for a given sentence according to a
    given grammar.
    """

    def __init__(self, sentence, grammar):
        self.words = sentence.split()
        self.grammar = grammar

        self.chart = Chart.init(len(self.words) + 1)

    def predictor(self, state, pos):
        """
        Earley Predictor.
        """

        for rule in self.grammar[state.next()]:
            self.chart[pos].add(State(rule, dot=0,
                sent_pos=state.chart_pos, chart_pos=state.chart_pos))

    def scanner(self, state, pos):
        """
        Earley Scanner.
        """

        if state.chart_pos < len(self.words):
            word = self.words[state.chart_pos]

            if any((word in r) for r in self.grammar[state.next()]):
                self.chart[pos + 1].add(State(Rule(state.next(), [word]),
                    dot=1, sent_pos=state.chart_pos,
                    chart_pos=(state.chart_pos + 1)))

    def completer(self, state, pos):
        """
        Earley Completer.
        """

        for prev_state in self.chart[state.sent_pos]:
            if prev_state.next() == state.rule.lhs:
                self.chart[pos].add(State(prev_state.rule,
                    dot=(prev_state.dot + 1), sent_pos=prev_state.sent_pos,
                    chart_pos=pos,
                    back_pointers=(prev_state.back_pointers + [state])))

    def parse(self):
        """
        Parses the sentence by running the Earley algorithm and filling out the
        chart.
        """

        # Checks whether the next symbol for the given state is a tag.
        def is_tag(state):
            return self.grammar.is_tag(state.next())

        for i in range(len(self.chart)):
            # print("Chart[" + str(i) + "]")
            for state in self.chart[i]:
                if not state.is_complete():
                    if is_tag(state):
                        self.scanner(state, i)
                    else:
                        self.predictor(state, i)
                else:
                    self.completer(state, i)
            # print("Num. of elems.:", len(self.chart[i]))
            # print(self.chart[i])
            # print("-" * 42)

    def has_parse(self):
        """
        Checks whether the sentence has a parse.
        """

        for state in self.chart[-1]:
            if state.is_complete() and state.rule.lhs == 'S' and \
                state.sent_pos == 0 and state.chart_pos == len(self.words):
                return True

        return False

    def get(self):
        """
        Returns the parse if it exists, otherwise returns None.
        """

        def get_helper(state):
            if self.grammar.is_tag(state.rule.lhs):
                return Tree(state.rule.lhs, [state.rule.rhs[0]])

            # if len(state.back_pointers) > 1:
            return Tree(state.rule.lhs,
                [get_helper(s) for s in state.back_pointers])
            # elif len(state.back_pointers) == 1:
            #     return get_helper(state.back_pointers[0])
            # else:
            #     return Tree(state.rule.lhs, [])

        for state in self.chart[-1]:
            if state.is_complete() and state.rule.lhs == 'S' and \
                state.sent_pos == 0 and state.chart_pos == len(self.words):
                return get_helper(state)

        return None


class Lexer():
    """
    Simple lexer for Python programs
    """

    def __init__(self):
        self.lexer = get_lexer_by_name("python")

    def lex(self, input_program):
        program = input_program
        if len(input_program) > 1:
            if input_program[-1] != '\n':
                program = input_program + '\n'
        program = self.remove_comments_and_strings(program)
        # Clean tabs
        all_lines = []
        for line in program.split('\n'):
            spaces_so_far = 0
            if len(line) > 0:
                if line[0] in [' ', '\t']:
                    for char in line:
                        if char == ' ':
                            spaces_so_far += 1
                        elif char == '\t':
                            spaces_so_far = (spaces_so_far // 4 + 1) * 4
                        else:
                            break
            all_lines.append(' ' * spaces_so_far + line.lstrip().replace('\t', '    '))
        all_lines = list(map(lambda line: list(pygments.lex(line.rstrip(), self.lexer)), all_lines))
        all_lines = self.update_indents_stack(all_lines)
        all_lines = self.update_spaces_and_nls(all_lines)
        all_lines = self.update_tokens(all_lines)
        tokens = [tok for line in all_lines for tok in line]
        tokens = self.final_cleaning(tokens)
        return tokens

    def remove_comments_and_strings(self, input_prog):
        prog = re.sub(re.compile(r"\\\s*?\n") , "" , input_prog)
        # Temporary replacements
        prog = prog.replace("\\\\", "__temporary__")
        prog = prog.replace("\\\"", "__double_quote__")
        prog = prog.replace("\\\'", "__single_quote__")
        prog = prog.replace("__temporary__", "\\\\")
        # String and comment replacements
        prog = re.sub(re.compile(r"\n\s*#.*?\n") , "\n" , prog)
        prog = re.sub(re.compile(r"\"\"\".*?\"\"\"", flags=re.DOTALL) , "__triple_dstring__" , prog)
        prog = re.sub(re.compile(r"\'\'\'.*?\'\'\'", flags=re.DOTALL) , "__triple_sstring__" , prog)
        in_double_quote = False
        in_single_quote = False
        in_comment = False
        new_prog = ""
        for char in prog:
            if not in_comment:
                if not in_double_quote and not in_single_quote and char == "#":
                    in_comment = True
                    new_prog += char
                elif not in_double_quote and not in_single_quote and char == "\"":
                    in_double_quote = True
                    new_prog += char
                elif not in_double_quote and not in_single_quote and char == "\'":
                    in_single_quote = True
                    new_prog += char
                elif in_double_quote and not in_single_quote and char == "\"":
                    in_double_quote = False
                    new_prog += char
                elif in_double_quote and not in_single_quote and char == "\'":
                    new_prog += "__single_quote__"
                elif not in_double_quote and in_single_quote and char == "\'":
                    in_single_quote = False
                    new_prog += char
                elif not in_double_quote and in_single_quote and char == "\"":
                    new_prog += "__double_quote__"
                else:
                    new_prog += char
            else:
                if char ==  "\n":
                    in_comment = False
                    new_prog += char
                elif char == "\'":
                    new_prog += "__single_quote__"
                elif char == "\"":
                    new_prog += "__double_quote__"
                else:
                    new_prog += char
        prog = new_prog
        prog = re.sub(re.compile(r"\"([^(\"|\'|\n)]|\(|\)|\|)*?\"") , "\"__string__\"" , prog)
        prog = re.sub(re.compile(r"\'([^(\"|\'|\n)]|\(|\)|\|)*?\'") , "\'__string__\'" , prog)
        prog = prog.replace("__triple_dstring__", "\"__string__\"")
        prog = prog.replace("__triple_sstring__", "\'__string__\'")
        prog = re.sub(re.compile(r"#.*?\n" ) , "\n" , prog)
        prog = re.sub(re.compile(r"\n\s+\n" ) , "\n" , prog)
        while prog.find('\n\n') >= 0:
            prog = prog.replace('\n\n', '\n')
        return prog

    def update_indents_stack(self, all_lines):
        all_line_tokens = []
        lst_token_prev_line = False
        fst_token_this_line = False
        indents = []
        paren_so_far = 0
        curly_so_far = 0
        square_so_far = 0
        for token_list in all_lines:
            fst_token = token_list[0]
            tok_idx = 0
            fst_real_token = token_list[tok_idx]
            while fst_real_token[0] in Text and fst_real_token[1].replace(' ', '') == '':
                tok_idx += 1
                if tok_idx < len(token_list):
                    fst_real_token = token_list[tok_idx]
                else:
                    break
            fst_token_this_line = fst_real_token[0] in Operator and fst_real_token[1] in ['+', '-', '*', '/', '//', '%',  '==', '!=', 'in', 'or', 'and'] and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            fst_token_this_line |= fst_real_token[0] in Punctuation and fst_real_token[1] in [',', '}', ')', ']'] and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            fst_token_this_line |= fst_real_token[0] in Punctuation and fst_real_token[1] in ['(', '['] and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            fst_token_this_line |= fst_real_token[0] in Keyword and fst_real_token[1] == 'for' and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            fst_token_this_line |= fst_real_token[0] in String and lst_token_prev_line and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            if lst_token_prev_line:
                # Checks if previous line ends with an operator, paren etc. and we are within a parenthesis, thus we must not indent
                last_line_tokens = all_line_tokens.pop()
                if len(last_line_tokens) > 1:
                    all_line_tokens.append(last_line_tokens[:-1])
                all_line_tokens.append(token_list)
            elif fst_token_this_line:
                # Checks if line starts with an operator and we are within a parenthesis, thus we must not indent
                last_line_tokens = all_line_tokens.pop()
                if len(last_line_tokens) > 1:
                    all_line_tokens.append(last_line_tokens[:-1])
                all_line_tokens.append(token_list[tok_idx:])
            elif fst_token[0] in Text and fst_token[1].replace(' ', '') == '':
                this_indent = len(fst_token[1])
                if indents == [] and this_indent > 0:
                    indents.append(this_indent)
                    all_line_tokens.append([(fst_token[0], '_INDENT_')] + token_list[1:])
                elif indents == []:
                    all_line_tokens.append(token_list[1:])
                elif indents[-1] == this_indent:
                    all_line_tokens.append(token_list[1:])
                elif indents[-1] < this_indent:
                    indents.append(this_indent)
                    all_line_tokens.append([(fst_token[0], '_INDENT_')] + token_list[1:])
                elif indents[-1] > this_indent:
                    dedents = 0
                    while indents[-1] > this_indent:
                        dedents += 1
                        indents.pop()
                        if indents == []:
                            break
                    all_line_tokens.append([(fst_token[0], '_DEDENT_')] * dedents + token_list[1:])
            elif not(fst_token[0] in Text and fst_token[1].replace('\n', '') == '') and \
                    len(indents) > 0:
                all_line_tokens.append([(Text, '_DEDENT_')] * len(indents) + token_list)
                indents = []
            else:
                all_line_tokens.append(token_list)
            if len(token_list) > 1:
                lst_token = token_list[-2]
                for tok in token_list:
                    if tok[0] in Punctuation and tok[1] == '(':
                        paren_so_far += 1
                    elif tok[0] in Punctuation and tok[1] == ')':
                        paren_so_far -= 1
                for tok in token_list:
                    if tok[0] in Punctuation and tok[1] == '{':
                        curly_so_far += 1
                    elif tok[0] in Punctuation and tok[1] == '}':
                        curly_so_far -= 1
                for tok in token_list:
                    if tok[0] in Punctuation and tok[1] == '[':
                        square_so_far += 1
                    elif tok[0] in Punctuation and tok[1] == ']':
                        square_so_far -= 1
                lst_token_prev_line = lst_token[0] in Punctuation and lst_token[1] in ['\\', '{', '(', '[']
                lst_token_prev_line |= lst_token[0] in Punctuation and lst_token[1] == ','  and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
                lst_token_prev_line |= token_list[-1][0] in Text and token_list[-1][1] == '\\\n'
                lst_token_prev_line |= lst_token[0] in Punctuation and lst_token[1] == ':' and curly_so_far > 0
                lst_token_prev_line |= lst_token[0] in Operator and lst_token[1] in ['+', '-', '*', '/', '//', '%', '==', '!=', 'in', 'or', 'and'] and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
                lst_token_prev_line |= lst_token[0] in String and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
        if len(indents) > 0:
            all_line_tokens.append([(Text, '_DEDENT_')] * len(indents))
        return all_line_tokens


    def update_spaces_and_nls(self, all_lines):
        def is_space(token):
            return token[0] in Text and token[1].replace(' ', '') == ''

        all_line_tokens = []
        for token_list in all_lines:
            token_list_no_spaces = list(filter(lambda tok: not is_space(tok), token_list))
            last_token = token_list_no_spaces[-1]
            if last_token[0] in Text and '\n' in last_token[1]:
                all_line_tokens.append(token_list_no_spaces[:-1] + [(last_token[0], '_NEWLINE_')])
            else:
                all_line_tokens.append(token_list_no_spaces)
        return all_line_tokens

    def update_tokens(self, all_lines):
        all_line_tokens = []
        for token_list in all_lines:
            new_token_list = []
            prev_num = False
            for tok in token_list:
                if tok[0] in Number:
                    prev_num = True
                else:
                    if prev_num and tok[0] in Name and tok[1] == 'j':
                        prev_tok = new_token_list.pop()
                        tok = (prev_tok[0], prev_tok[1] + 'j')
                    prev_num = False
                new_token_list.append(tok)
            new_token_list = list(map(self.choose_token_represent, new_token_list))
            all_line_tokens.append(new_token_list)
        return all_line_tokens

    def choose_token_represent(self, token):
        if token[0] in Name and token[1] != '.':
            return '_NAME_'
        elif token[0] in Number:
            return '_NUMBER_'
        elif token[0] in String:
            return '_STRING_'
        return token[1]

    def final_cleaning(self, tokens):
        tokens.append('_ENDMARKER_')
        tokens = " ".join(tokens)
        tokens = tokens.replace('* *', "**")
        tokens = tokens.replace('= =', "==")
        tokens = tokens.replace('< =', "<=")
        tokens = tokens.replace('> =', ">=")
        tokens = tokens.replace('! =', "!=")
        tokens = tokens.replace('< <', "<<")
        tokens = tokens.replace("> >", ">>")
        tokens = tokens.replace('& &', "&&")
        tokens = tokens.replace('| |', "||")
        tokens = tokens.replace('/ /', "//")
        tokens = tokens.replace('+ =', "+=")
        tokens = tokens.replace('- =', "-=")
        tokens = tokens.replace('/ =', "/=")
        tokens = tokens.replace('* =', "*=")
        tokens = tokens.replace('>> =', ">>=")
        tokens = tokens.replace('<< =', "<<=")
        tokens = tokens.replace('&& =', "&&=")
        tokens = tokens.replace('!! =', "!!=")
        tokens = tokens.replace('// =', "//=")
        tokens = tokens.replace('% =', "%=")
        tokens = tokens.replace('@', "@ ")
        tokens = tokens.replace('@ =', "@=")
        tokens = tokens.replace('| =', "|=")
        tokens = tokens.replace('& =', "&=")
        tokens = tokens.replace('^ =', "^=")
        # tokens = tokens.replace(", ;", ";")
        tokens = tokens.replace(". . .", "...")
        tokens = tokens.replace("not in", "not_in")
        tokens = tokens.replace("is not", "is_not")
        tokens = tokens.replace("- >", "_arrow_")
        while tokens.find('_STRING_ _STRING_') >= 0:
            tokens = tokens.replace('_STRING_ _STRING_', '_STRING_')
        while tokens.find(' : _NEWLINE_ _NEWLINE_ ') >= 0:
            tokens = tokens.replace(' : _NEWLINE_ _NEWLINE_ ', ' : _NEWLINE_ ')
        while tokens.find('. _NUMBER_') >= 0:
            tokens = tokens.replace('. _NUMBER_', '_NUMBER_')
        while tokens.find('_NEWLINE_ )') >= 0:
            tokens = tokens.replace('_NEWLINE_ )', ')')
        while tokens.find('_NEWLINE_ ]') >= 0:
            tokens = tokens.replace('_NEWLINE_ ]', ']')
        while tokens.find('_NEWLINE_ }') >= 0:
            tokens = tokens.replace('_NEWLINE_ }', '}')
        # print(tokens.replace('_NEWLINE_ ', '\n'))
        # tokens = " ".join(map(lambda t: t if t in self.terminals else '_UNKNOWN_', tokens.split()))
        # print(tokens.replace('_NEWLINE_ ', '\n'))
        return tokens


def read_grammar(grammar_file):
    grammar = Grammar.load_grammar(grammar_file)
    return grammar


def prog_has_parse(prog, grammar):
    def run_parse(sentence):
        parser = EarleyParse(sentence, grammar)
        parser.parse()
        return parser.has_parse(), parser.chart

    lexer = Lexer()
    tokenized_prog = lexer.lex(prog)
    # print('-----------------')
    # print(tokenized_prog.replace('_NEWLINE_ ', '\n'))
    # print('-----------------')
    parsed, _ = run_parse(tokenized_prog)
    if parsed is None:
        return False
    else:
        return parsed


def main():
    """
    Main.
    """

    parser_description = ("Runs the Earley parser according to a given "
        "grammar.")

    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument('draw', nargs='?', default=False)
    parser.add_argument('grammar_file', help="Filepath to grammer file")
    parser.add_argument('input_program', help="The input program to parse")
    parser.add_argument('--show-chart', action="store_true")

    args = parser.parse_args()

    grammar = Grammar.load_grammar(args.grammar_file)

    def run_parse(sentence):
        parser = EarleyParse(sentence, grammar)
        parser.parse()
        return parser.get(), parser.chart

    program_path = Path(args.input_program)
    input_program = program_path.read_text()

    lexer = Lexer()
    tokenized_prog = lexer.lex(input_program)
    # print(parse(input_program))
    print(tokenized_prog.replace('_NEWLINE_ ', '\n'))
    print('-----------------')
    parsed, chart = run_parse(tokenized_prog)
    if args.show_chart:
        print(chart)
        print('\n')
    if parsed is None:
        print(input_program + '\n')
    else:
        if args.draw:
            parsed.draw()
        else:
            print("True")
            # parsed.pretty_print()


if __name__ == '__main__':
    main()
