from compartor.compartments import Content, ContentChange, Compartment, Transition
from compartor.compartments import Moment, Bulk
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

def _parse_bulk(expr):
    pass

def _parse(expr):
    if expr is EmptySet:
        return EmptySet, False
    elif type(expr) in [tuple, dict] and len(expr) == 0:
        return EmptySet, False
    elif type(expr) is list:
        return Add(*[_parse_content(c) for c in expr]), False
    elif type(expr) is set: # {x} bulk syntax
        return Add(*[_parse_content(c) for c in expr]), True
    # elif expr.func in [Moment,Add]:
    #     return expr, True
    else:
        raise TypeError("Unexpected:" + str(expr))


def _make_transition(reactants, products):
    R, isBulkR = _parse(reactants)
    P, isBulkP = _parse(products)
    if not isBulkR and not isBulkP:
        return Transition(R, P)
    elif isBulkR and isBulkP:
        return Transition(R, P, isBulk=True)
    else:
        raise TypeError("Transition must be either Compartmental or Bulk, not mixed!")
    return None


to = _Infix(_make_transition)
