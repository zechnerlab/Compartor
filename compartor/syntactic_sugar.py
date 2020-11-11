from compartor.compartments import Content, ContentChange, Compartment, Transition
from sympy import EmptySet, Add


class _Infix:
    def __init__(self, function):
        self.function = function

    def __rlshift__(self, other):
        return _Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rrshift__(self, other):
        return _Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rsub__(self, other):
        return _Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __gt__(self, other):
        return self.function(other)

    def __neg__(self):
        return self

    def __call__(self, value1, value2):
        return self.function(value1, value2)


def _parse_content(expr):
    if type(expr) is tuple:
        return Compartment(ContentChange(*expr))
    else:
        return Compartment(expr)


def _parse(expr):
    if expr is EmptySet:
        return EmptySet
    elif type(expr) in [tuple, dict] and len(expr) == 0:
        return EmptySet
    elif type(expr) is list:
        return Add(*[_parse_content(c) for c in expr])
    else:
        raise TypeError("Unexpected:" + str(expr))


def _make_transition(reactants, products):
    return Transition(_parse(reactants), _parse(products))


to = _Infix(_make_transition)
