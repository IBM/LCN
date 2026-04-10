# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The LCN parser (i.e., used to parse propositional logical formulas)

import re
from .json_schema import JsonPropositionalLogicSchema

"""
Propositional Logic Clause Parser
"""

BOOLEAN_FALSE = 0
BOOLEAN_TRUE = 1
NOT_OPERATOR = -1
AND_OPERATOR = -2
XOR_OPERATOR = -3
OR_OPERATOR = -4
NAND_OPERATOR = -5
XNOR_OPERATOR = -6
NOR_OPERATOR = -7

def xor(a):
    # parity check
    return len(list(filter(lambda x: x, a))) % 2 != 0

# https://en.wikipedia.org/wiki/List_of_logic_symbols
OPERATORS = {
    # https://en.wikipedia.org/wiki/Negation
    NOT_OPERATOR: {'word': 'not', 'char': '!', 'math': '¬', 'func': lambda x: not x},
    # https://en.wikipedia.org/wiki/Logical_conjunction
    AND_OPERATOR: {'word': 'and', 'char': '&', 'math': '∧', 'func': all},
    # https://en.wikipedia.org/wiki/Exclusive_or
    XOR_OPERATOR: {'word': 'xor', 'char': '^', 'math': '⊕', 'func': xor},
    # https://en.wikipedia.org/wiki/Logical_disjunction
    OR_OPERATOR:  {'word': 'or', 'char': '|', 'math': '∨', 'func': any},
    # https://en.wikipedia.org/wiki/Sheffer_stroke
    NAND_OPERATOR: {'word': 'nand', 'char': '/', 'math': '↑', 'func': lambda a: not all(a)},
    # https://en.wikipedia.org/wiki/Logical_biconditional
    XNOR_OPERATOR: {'word': 'xnor', 'char': '=', 'math': '↔', 'func': lambda a: not xor(a)},
    # https://en.wikipedia.org/wiki/Logical_NOR
    NOR_OPERATOR:  {'word': 'nor', 'char': '†', 'math': '↓', 'func': lambda a: not any(a)}
}

BOOLEANS = {
    BOOLEAN_TRUE: {'word': 'true', 'char': '1', 'math': '⊤'},
    BOOLEAN_FALSE: {'word': 'false', 'char': '0', 'math': '⊥'}
}

TRUES = [BOOLEANS[BOOLEAN_TRUE]['word'], BOOLEANS[BOOLEAN_TRUE]['char'], BOOLEANS[BOOLEAN_TRUE]['math']]

class ParseException(Exception):
    pass

class FormulaParser():

    def __init__(
            self, 
            parentheses = ['(', ')'], 
            wrappers = ["'", '"'], 
            operator_precedence = (-7, -6, -5, -4, -3, -2, -1)
    ):
        """ 
        PLCParser class constructor 
        
        Args:
            parantheses: list
                The allowed parantheses in the formula
            wrappers: list
                String wrappers
            operator_precedence: list
                The precedence of the operators
            
        """
        self.OPEN_PARENTHESES, self.CLOSE_PARENTHESES = parentheses
        # http://stackoverflow.com/questions/430759/regex-for-managing-escaped-characters-for-items-like-string-literals
        self.wrappers = wrappers
        self.STRING_LITERALS = re.compile('|'.join([r"%s[^%s\\]*(?:\\.[^%s\\]*)*%s" % 
                                          (w,w,w,w) for w in self.wrappers]))
        self.operator_precedence = operator_precedence
        self.operator_schemas = {}
        self.negate_unary_operator = False
    
    def set_literals(
            self, 
            input_string: str
    ) -> None:
        """
        Identify the literals in the input formula.

        Args:
            input_string: str
                The input logical formula (propositional logic)
        """
        
        self.literals = {}
        # find literals
        lit = self.STRING_LITERALS.search(input_string)
        n = 1
        while lit:
            g = lit.group()
            key = "L%s" % n
            # set literal by key and value by removing 
            # for example " and ' wrapper chars
            self.literals[key] = g[1:-1]
            # remove literal from original input and replace with key
            input_string = input_string.replace(g, " %s " % key)
            # make a new search and loop until all is found
            lit = self.STRING_LITERALS.search(input_string)
            # next literal number
            n += 1
        # wrap parenthesis and operator symbols with space for later usage
        input_string = input_string.replace(self.OPEN_PARENTHESES, ' %s ' % self.OPEN_PARENTHESES)
        input_string = input_string.replace(self.CLOSE_PARENTHESES, ' %s ' % self.CLOSE_PARENTHESES)
        for operator, options in OPERATORS.items():
            input_string = input_string.replace(options['math'], ' %s ' % options['math'])
            input_string = input_string.replace(options['char'], ' %s ' % options['char'])
        # set literal string and its length for recursive parser
        self.literal_string = input_string
      
    def _parse(
            self, 
            l, 
            operators, 
            unary_operator
    ):
        """
        Helper function for parsing the formula.

        Args:
            l: list
                A list containing the formula objects
            operators: list
                The logical operators
            unary_operators: list
                The unary logical operators
        """
        
        # http://stackoverflow.com/questions/42032418/group-operands-by-logical-connective-precedence-from-python-list
        # if nothing on list, raise error
        if len(l) < 1:
            raise ParseException("Malformed input. Length 0 or multiple operators.")
        # one item on list
        if len(l) == 1:
            # if not negation or other operators
            if l[0] != unary_operator and not l[0] in operators:
                # substitute literals back to original content if available
                if not isinstance(l[0], list) and l[0] in self.literals:
                    l[0] = self.literals[l[0]]
                    # finally replace escaped content with content
                    for w in self.wrappers:
                        l[0] = l[0].replace("\\%s" % w, w)
                # return data
                return l[0]
            # raise error because operator should not exist at this point
            else:
                raise ParseException("Malformed input. Operator not found: %s." % l[0])
        # we are looping over all binary operators in order: 
        # -4 = or, -3 = xor, -2 = and
        if len(operators) > 0:
            operator = operators[0]
            try:
                # for left-associativity of binary operators
                position = len(l) - 1 - l[::-1].index(operator)
                return [self._parse(l[:position], operators, unary_operator), operator, self._parse(l[position+1:], operators[1:], unary_operator)]
            except ValueError:
                return self._parse(l, operators[1:], unary_operator)
        # expecting only not operator at this point
        if l[0] != unary_operator:
            raise ParseException("Malformed input. Operator expected, %s found." % l[0])
        # return data with not operator
        return [unary_operator, self._parse(l[1:], operators, unary_operator)]

    def substitute(
            self, 
            x
    ):
        """
        """
        
        y = x.strip().lower()
        # operators
        for op, d in OPERATORS.items():
            if y == d['word'] or y == d['char'] or y == d['math']:
                return op
        # booleans
        for op, d in BOOLEANS.items():
            if y == d['word'] or y == d['char'] or y == d['math']:
                return op
        # other
        return x

    def recursive_parentheses_groups(self):
        """
        Identify recursively the groups of parantheses in the formula.
        """
        
        def rec(a):
            # http://stackoverflow.com/questions/17140850/how-to-parse-a-string-and-return-a-nested-array/17141899#17141899
            stack = [[]]
            for x in a:
                if x == self.OPEN_PARENTHESES:
                    stack[-1].append([])
                    stack.append(stack[-1][-1])
                elif x == self.CLOSE_PARENTHESES:
                    stack.pop()
                    # special treatment for prefix notation
                    # (^ a b c) => (a ^ b ^ c) => (a ^ (b ^ c))
                    # (& a b c) => (a & b & c) => (a & (b & c))
                    # (| a b c) => (a | b | c) => (a | (b | c))
                    # also (& a b (| c d)) => (a & b) & (c | d) is possible!
                    if len(stack[-1][-1]) > 1 and \
                       isinstance(stack[-1][-1][0], int) and \
                       stack[-1][-1][0] in OPERATORS and \
                       stack[-1][-1][0] != NOT_OPERATOR:
                        op = stack[-1][-1][0]
                        b = []
                        # append operator between all operands
                        for a in stack[-1][-1][1:]:
                            b.extend([a, op])
                        # in special case where only one operand and one operator is found
                        # we should handle the case differently on evaluation process
                        # TODO: it is not possible to deformat this kind of structure
                        # back to string representation until deformat method is extended...
                        if op < -4 and len(b) == 2:
                            self.negate_unary_operator = True
                        stack[-1][-1] = b[:-1]
                    # see if remaining content has more than one
                    # operator and make them nested set in that case
                    # last operator predecende is usually NOT_OPERATOR
                    # but can be configured at the object initialization
                    stack[-1][-1] = self._parse(stack[-1][-1], self.operator_precedence[:-1], self.operator_precedence[-1])
                    if not stack:
                        raise ParseException('Malformed input. Parenthesis mismatch. Opening parentheses missing.')
                else:
                    stack[-1].append(x)
            if len(stack) > 1:
                raise ParseException('Malformed input. Parenthesis mismatch. Closing parentheses missing.')
            return stack.pop()

        # remove whitespace from literal string (!=input string at this point already)
        a = ' '.join(self.literal_string.split()).split(' ')
        # artificially add parentheses if not provided
        a = [self.OPEN_PARENTHESES] + a + [self.CLOSE_PARENTHESES]
        # substitute different operators by numeral representatives
        a = map(self.substitute, a)
        # loop over the list of literals placeholders and operators and parentheses
        return rec(list(a))

    def parse(
            self, 
            input_string: str
    ):
        """
        Main parsing method.

        Args:
            input_string: str
                The input propositional logic formula.

        Returns:
            The parse tree of the formula (as a list of lists)
        """
        
        # first set literals
        self.set_literals(input_string.strip())
        # then recursively parse clause
        return self.recursive_parentheses_groups()[0]

    @staticmethod
    def parse_input(input_string):
        """ 
        Parse the input propositional logic formula
        
        Args:
            input_string: str
                A string representing the formula
        
        Returns:
            A tuple containing the parsed formula and its variables
        """
        c = FormulaParser()
        try:
            o = c.parse(input_string)

            def flatten_list(nested_list):
                def flatten(lst):
                    for item in lst:
                        if isinstance(item, list):
                            flatten(item)
                        else:
                            flat_list.append(item)

                flat_list = []
                flatten(nested_list)
                return flat_list

            i = 1
            parsed_variables = {}
            for elem in flatten_list([o]):
                if isinstance(elem, str): # variable
                    if elem not in parsed_variables:
                        parsed_variables[elem] = f"V{i}"
                        i = i + 1

            return o, {v: k for k, v in parsed_variables.items()}
        except ParseException as pe:
            return None, None

    @staticmethod
    def validate_input(input_string):
        """ bypass object construct """
        c = FormulaParser()
        try:
            return c.validate(input_string)
        except:
            return None
    
    def validate(
            self, 
            input_string, 
            open_parenthesis=None, 
            close_parenthesis=None, 
            wrappers=[], 
            escape_char=None
    ):
        """
        Validate the logical formula (check if it is well formed syntactically)

        Args:
            input_string: str
                The input logical formula
            open_paranthesis: list
                The type of open paranthesis (e.g., (, [, {)
            closed_paranthesis: list
                The type of closed paranthesis (e.g., ), ], })
            wrappers: list
                The type of string wrappers
            escape_char: list
                The type of escape characters
        """
        
        # check parentheses and wrappers characters that they match
        # for example (, [, {
        open_parenthesis = open_parenthesis if open_parenthesis else self.OPEN_PARENTHESES
        # for example: }, ], )
        close_parenthesis = close_parenthesis if close_parenthesis else self.CLOSE_PARENTHESES
        # multiple wrapper chars accepted, for example ['"', "'", "´"]
        wrappers = wrappers if wrappers else self.wrappers
        # is is possible to pass a different escape char, but it is probably
        # not a good idea because many of the string processors use the same
        escape_char = escape_char if escape_char else '\\'
        # init vars
        stack, current, last, previous = ([], None, None, None)
        # loop over all characters in a string
        for current in input_string:
            #if previous character was escape character, then 
            # swap it with the current one and continue to the next char
            if previous == escape_char:
                # see if current character is escape char, then there are
                # two of them in row and we should reset previous marker
                if current == escape_char:
                    previous = None
                else:
                    previous = current
                continue
            # last stacked char. not that this differs from the previous value which
            # is the previous char from string. last is the last char from stack
            last = stack[-1] if stack else None
            # if we are inside a wrapper accept ANY character 
            # until the next unescaped wrapper char occurs
            if last in wrappers and current != last:
                # swap the current so that we can escape wrapper inside wrappers: "\""
                previous = current
                continue
            # push open parenthesis or wrapper to the stack
            if current == open_parenthesis or (current in wrappers and current != last):
                stack.append(current)
            # prepare to pop last parenthesis or wrapper
            elif current == close_parenthesis or current in wrappers:
                # if there is nothing on stack, should already return false
                if len(stack) == 0:
                    return False
                else:
                    # if we encounter wrapper char take the last wrapper char out from stack
                    # or if the last char was open and current close parenthsis
                    if last in wrappers or (last == open_parenthesis and current == close_parenthesis):
                        stack.pop()
                    else:
                        return False
            # update previous char
            previous = current
        # if there is something on stack then no closing char was found
        return len(stack) == 0

    @staticmethod
    def deformat_input(lst, operator_type="word"):
        """ bypass object construct. Types are: word, char, math"""
        c = FormulaParser()
        try:
            return c.deformat(lst, operator_type=operator_type)
        except:
            return None
    
    def deformat(self, lst, operator_type = "word"):

        def rec(current_item, operator_type, first=True):
            # if item is not a list, return value
            if not isinstance(current_item, list):
                # boolean values
                y = str(current_item).lower()
                for op, d in BOOLEANS.items():
                    if y == d['word'] or y == d['char'] or y == d['math']:
                        return current_item
                # normal items wrapped by the first configured wrapper char
                # escaping wrapper char inside the string
                current_item = current_item.replace(self.wrappers[0], '\\'+self.wrappers[0])
                return self.wrappers[0]+current_item+self.wrappers[0]
            # item is a list, open clause
            a = []
            if not first:
                a.append(self.OPEN_PARENTHESES)
            # loop all items
            for item in current_item:
                # operators
                if not isinstance(item, list) and item in OPERATORS:
                    a.append(OPERATORS[item][operator_type])
                # item or list
                else:
                    # recursively add next items
                    a.append(rec(item, operator_type, False))
            # close clause
            if not first:
                a.append(self.CLOSE_PARENTHESES)
            return ' '.join(map(str, a))

        # call sub routine to deformat structure
        return rec(lst, operator_type)

    @staticmethod
    def evaluate_input(i, table={}):
        """ bypass object construct """
        c = FormulaParser()
        try:
            return c.evaluate(i, table)
        except:
            return None

    def evaluate(
            self, 
            i, 
            table={}
    ):
        """
        Evaluate the truth value of a formula.

        Returns:
            The truth value (True, False) of the formula, given the truth
            assignments to its literals.
        """
        
        # parse input if string, otherwise expecting correctly structured list
        return self.truth_value(self.parse(i) if type(i) == str else i, table)

    def truth_value(
            self, 
            current_item, 
            table, 
            negate=True
    ):
        """
        Get the truth value of the current item in the formula.

        Args:
            current_item: list
                The current item in the formula to be evaluated
            table: dict
                The truth assignments of the literals
            negate: boolean
                Flag indicating whether to negate or not the result
        """
        
        # if item is not a list, check the truth value
        if not isinstance(current_item, list):
            # truth table is possibly given
            if table and current_item in table:
                current_item = table[current_item]
            # force item to string and lowercase for simpler comparison
            # if only single operator is given on input, then self.negate_unary_operator
            # is set to true, thus comparison here is done
            return (str(current_item).lower() in TRUES) != self.negate_unary_operator
        # item is a list
        a = []
        # default operator
        operator = AND_OPERATOR
        for item in current_item:
            # operators
            if not isinstance(item, list) and item in OPERATORS:
                if item == NOT_OPERATOR:
                    negate = False
                else:
                    operator = item
            else:
                a.append(self.truth_value(item, table))
        # all operators have a function to check the truth value
        # we must compare returned boolean against negate parameter
        return OPERATORS[operator]['func'](a) == negate

    @staticmethod
    def get_variables(input_string):
        """ bypass object construct """
        c = FormulaParser()
        try:
            o = c.parse(input_string)

            def flatten_list(nested_list):
                def flatten(lst):
                    for item in lst:
                        if isinstance(item, list):
                            flatten(item)
                        else:
                            flat_list.append(item)

                flat_list = []
                flatten(nested_list)
                return flat_list

            i = 1
            parsed_variables = {}
            for elem in flatten_list(o):
                if isinstance(elem, str): # variable
                    if elem not in parsed_variables:
                        parsed_variables[elem] = f"V{i}"
                        i = i + 1
            return {v: k for k, v in parsed_variables.items()}
        except:
            return None

    @staticmethod
    def json_schema(i, table={}):
        """ bypass object construct """
        c = FormulaParser()
        try:
            return c.build_json_schema(i, table)
        except:
            return None

    def build_json_schema(self, i, table={}):
        schema = self.schema(self.parse(i) if type(i) == str else i, table)
        # after running schema method all required schema components are collected
        ## and returned as a json string
        jpls = JsonPropositionalLogicSchema()
        return jpls.get(schema, **self.operator_schemas)

    def schema(
            self, 
            current_item, 
            table, 
            negate=True
    ):
        """
        Get the JSON schema of the formula.

        Args:
            current_item: list
                The current item in the formula
            table: dict
                The truth assignments (if any) of the literals
            negate: boolean
                Flag indicating whether to negate the result or not
        """
        
        # item is a list
        a = []
        # should we negate next item, was it a list or values
        operator = -2
        for item in current_item:
            # operators
            if not isinstance(item, list) and item in OPERATORS:
                if item == NOT_OPERATOR:
                    negate = False
                else:
                    operator = item
            else:
                if isinstance(item, list):
                    a.append(self.schema(item, table))
        # is group AND / OR / XOR
        # take care of negation for the list result too
        if operator == XOR_OPERATOR or operator == XNOR_OPERATOR:
            # if any of the values are true, but not all
            if operator == XNOR_OPERATOR:
                op = '_xnor' if negate else '_xor'
            op = '_xor' if negate else '_xnor'
        elif operator == OR_OPERATOR or operator == NOR_OPERATOR:
            # if some of the values is true
            if operator == NOR_OPERATOR:
                op = '_nor' if negate else '_or'
            op = '_or' if negate else '_nor'
        # operator == AND_OPERATOR
        else:
            # if all values are true
            if operator == NAND_OPERATOR:
                op = '_nand' if negate else '_and'
            op = '_and' if negate else '_nand'
        # add used operator to the schema
        self.operator_schemas[op] = True
        # generate schema
        if len(a) > 0:
            items = '{"items": {%s}}' % ',\r\n\t'.join(a)
            return '{"allOf": [\r\n\t{"$ref": "#/definitions/%s"},\r\n\t%s]}' % (op, items)
        # the deep most nested level has no appended items on a
        else:
            return '"$ref": "#/definitions/%s"' % op

parse_formula = FormulaParser.parse_input
evaluate_formula = FormulaParser.evaluate_input
deformat_formula = FormulaParser.deformat_input
validate_formula = FormulaParser.validate_input
get_variables = FormulaParser.get_variables
json_schema = FormulaParser.json_schema