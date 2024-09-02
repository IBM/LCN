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

# Independencies in an LCN model given by the local/globa Markov conditions

class Independencies(object):
    """
    Base class for independencies. The Independencies class represents a set of 
    Conditional Independence assertions (e.g.,: "X is conditionally independent 
    of Y given Z" where X, Y and Z are random variables) or Independence 
    assertions (eg: "X is independent of Y" where X and Y are random variables).
    Initialize the independencies Class with Conditional Independence
    assertions or Independence assertions.

    Args:
        assertions: Lists or Tuples
            Each assertion is a list or tuple of the form: [event1, event2 and 
            event3] e.g.,: assertion ['X', 'Y', 'Z'] would be X is independent
            of Y given Z.

    Examples:
    ---------

    Creating an independencies object with one independence assertion:
    Random Variable X is independent of Y

    independencies = independencies(['X', 'Y'])

    Creating an independencies object with three conditional
    independence assertions: random variable X is independent of Y given Z.

    independencies = independencies(['X', 'Y', 'Z'],
                ['a', ['b', 'c'], 'd'],
                ['l', ['m', 'n'], 'o'])

    """

    def __init__(self, *assertions):
        self.independencies = []
        self.add_assertions(*assertions)

    def __str__(self):
        string = "\n".join([str(assertion) for assertion in self.independencies])
        return string

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Independencies):
            return False
        return all(
            independency in other.get_assertions()
            for independency in self.get_assertions()
        ) and all(
            independency in self.get_assertions()
            for independency in other.get_assertions()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def contains(self, assertion):
        """
        Returns `True` if `assertion` is contained in this `Independencies`-object,
        otherwise `False`.

        Args:
            assertion: IndependenceAssertion()-object

        Examples:
        ---------
        ind = Independencies(['A', 'B', ['C', 'D']])
        IndependenceAssertion('A', 'B', ['C', 'D']) in ind
        True
        # does not depend on variable order:
        IndependenceAssertion('B', 'A', ['D', 'C']) in ind
        True
        """
        if not isinstance(assertion, IndependenceAssertion):
            raise TypeError(
                f"' in <Independencies()>' requires IndependenceAssertion as left operand, not {type(assertion)}"
            )

        return assertion in self.get_assertions()

    __contains__ = contains

    def get_all_variables(self):
        """
        Returns a set of all the variables in all the independence assertions.
        """
        return frozenset().union(*[ind.all_vars for ind in self.independencies])

    def get_assertions(self):
        """
        Returns the independencies object which is a set of IndependenceAssertion objects.

        Examples:
        --------
        independencies = Independencies(['X', 'Y', 'Z'])
        independencies.get_assertions()

        """
        return self.independencies

    def add_assertions(self, *assertions):
        """
        Adds assertions to independencies.

        Args:
            assertions: Lists or Tuples
                Each assertion is a list or tuple of variable, independent_of and given.

        Examples:
        --------
        independencies = Independencies()
        independencies.add_assertions(['X', 'Y', 'Z'])
        independencies.add_assertions(['a', ['b', 'c'], 'd'])
        """
        for assertion in assertions:
            if isinstance(assertion, IndependenceAssertion):
                self.independencies.append(assertion)
            else:
                try:
                    self.independencies.append(
                        IndependenceAssertion(assertion[0], assertion[1], assertion[2])
                    )
                except IndexError:
                    self.independencies.append(
                        IndependenceAssertion(assertion[0], assertion[1])
                    )


    def latex_string(self):
        """
        Returns a list of string.
        Each string represents the IndependenceAssertion in latex.
        """
        return [assertion.latex_string() for assertion in self.get_assertions()]


class IndependenceAssertion(object):
    """
    Represents Conditional Independence or Independence assertion. Each 
    assertion has 3 attributes: event1, event2, event3.
    
    The attributes for

    .. math:: U \perp X, Y | Z

    is read as: Random Variable U is independent of X and Y given Z would be:

    event1 = {U}

    event2 = {X, Y}

    event3 = {Z}

    Args:
        event1: String or List of strings
            Random Variable which is independent.

        event2: String or list of strings.
            Random Variables from which event1 is independent

        event3: String or list of strings.
            Random Variables given which event1 is independent of event2.

    Examples
    --------
    assertion = IndependenceAssertion('U', 'X')
    assertion = IndependenceAssertion('U', ['X', 'Y'])
    assertion = IndependenceAssertion('U', ['X', 'Y'], 'Z')
    assertion = IndependenceAssertion(['U', 'V'], ['X', 'Y'], ['Z', 'A'])

    """

    def __init__(self, event1=[], event2=[], event3=[]):
        """
        Initialize an IndependenceAssertion object with event1, event2 and event3 attributes.

                    event2
                    ^
        event1     /   event3
           ^      /     ^
           |     /      |
          (U || X, Y | Z) read as Random variable U is independent of X and Y given Z.
            ---
        """
        if event1 and not event2:
            raise ValueError("event2 needs to be specified")
        if any([event2, event3]) and not event1:
            raise ValueError("event1 needs to be specified")
        if event3 and not all([event1, event2]):
            raise ValueError(
                "event1" if not event1 else "event2" + " needs to be specified"
            )

        self.event1 = frozenset(self._return_list_if_not_collection(event1))
        self.event2 = frozenset(self._return_list_if_not_collection(event2))
        self.event3 = frozenset(self._return_list_if_not_collection(event3))
        self.all_vars = frozenset().union(self.event1, self.event2, self.event3)

    def __str__(self):
        if self.event3:
            return "({event1} \u27C2 {event2} | {event3})".format(
                event1=", ".join([str(e) for e in self.event1]),
                event2=", ".join([str(e) for e in self.event2]),
                event3=", ".join([str(e) for e in self.event3]),
            )
        else:
            return "({event1} \u27C2 {event2})".format(
                event1=", ".join([str(e) for e in self.event1]),
                event2=", ".join([str(e) for e in self.event2]),
            )

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, IndependenceAssertion):
            return False
        return (self.event1, self.event2, self.event3) == other.get_assertion() or (
            self.event2,
            self.event1,
            self.event3,
        ) == other.get_assertion()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((frozenset((self.event1, self.event2)), self.event3))

    @staticmethod
    def _return_list_if_not_collection(event):
        """
        If variable is a string returns a list containing variable.
        Else returns variable itself.
        """
        if isinstance(event, str):
            return [event]
        else:
            return event

    def get_assertion(self):
        """
        Returns a tuple of the attributes: variable, independent_of, given.

        Examples
        --------
        asser = IndependenceAssertion('X', 'Y', 'Z')
        asser.get_assertion()
        """
        return self.event1, self.event2, self.event3

    def latex_string(self):
        if len(self.event3) == 0:
            return r"{event1} \perp {event2}".format(
                event1=", ".join([str(e) for e in self.event1]),
                event2=", ".join([str(e) for e in self.event2]),
            )
        else:
            return r"{event1} \perp {event2} \mid {event3}".format(
                event1=", ".join([str(e) for e in self.event1]),
                event2=", ".join([str(e) for e in self.event2]),
                event3=", ".join([str(e) for e in self.event3]),
            )
