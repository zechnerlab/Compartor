from sympy import *
import collections
import itertools

# -------------------------------------------------
def ContentVar(name):
    return IndexedBase(name, integer=True, shape=1)

# -------------------------------------------------
class Compartment(Function):
    nargs = 1

    def __str__(self):
        return f'[{self.args[0]}]'

    def _sympystr(self, printer=None):
        return f'[{self.args[0]}]'

    def _latex(self, printer=None):
        return '\\left[' + printer.doprint(self.args[0]) + '\\right]'

    def content(self):
        return self.args[0]

# -------------------------------------------------
class CompartmentSum(Expr):
    def __init__(self, expr, var):
        self.expr = expr
        self.var = var

    def __str__(self):
        return f'Sum({self.expr}), ({self.var} ∈ \U0001D54F)'

    def _latex(self, printer=None):
        # Always use printer.doprint() otherwise nested expressions won't
        # work. See the example of ModOpWrong.
        lexpr = printer.doprint(self.expr)
        lvar = printer.doprint(self.var)
        return '\sum_{' + lvar + ' \in \mathbb{X}}' + lexpr


# -------------------------------------------------
__numCompartments = Function('n', integer=True)

def n(content):
    if content.func == Compartment:
        return n(content.args[0])
    return __numCompartments(content)

# -------------------------------------------------
def __kronecker(content1, content2):
    if content1.func == Compartment:
        return __kronecker(content1.args[0], content2)
    if content2.func == Compartment:
        return __kronecker(content1, content2.args[0])
    return KroneckerDelta(content1, content2)

# -------------------------------------------------
def __checkSimpleCompartment(expr):
    """Checks that expr is a Compartment(IndexedBase) and throws TypeError if it is not"""
    if not (expr.func == Compartment and len(expr.args) == 1 and expr.args[0].func == IndexedBase):
        raise TypeError(
            "Only compartments comprising a singe content variable are supported (not '" + str(expr) + "')")

# -------------------------------------------------
def __getMassAction(compartments):
    """
    Build w(n;Xc)

    Can handle [x], n*[x], and [x] + [y], where n is an integer, and x, y are compartment content variables

    :param dict compartments: maps Compartment to number of occurrences
    :return: mass action term w(n;Xc)
    """
    if len(compartments) == 0:
        return 1
    elif len(compartments) == 1:
        (compartment, count) = next(iter(compartments.items()))
        __checkSimpleCompartment(compartment)
        return 1 / factorial(count) * ff(n(compartment), count)
    elif len(compartments) == 2:
        i = iter(compartments.items())
        (compartment1, count1) = next(i)
        (compartment2, count2) = next(i)
        __checkSimpleCompartment(compartment1)
        __checkSimpleCompartment(compartment2)
        if count1 != 1 or count2 != 1:
            raise RuntimeError("Higher than 2nd order transitions are not implemented yet")
        return n(compartment1) * (n(compartment2) - __kronecker(compartment1, compartment2))
    else:
        raise RuntimeError("Higher than 2nd order transitions are not implemented yet")

# -------------------------------------------------
def getSumMassAction(compartments, expr):
    """
    Get sum_Xc(w(n;Xc)*expr)
    :param dict compartments: reactant compartments Xc as a dictionary that maps Compartment to number of occurrences
    :param expr: expression in the sum
    :return: sum_Xc(w(n;Xc)*expr)
    """
    if len(compartments) == 0:
        return expr
    elif len(compartments) == 1:
        (compartment, count) = next(iter(compartments.items()))
        __checkSimpleCompartment(compartment)
        w = 1 / factorial(count) * ff(n(compartment), count)
        return CompartmentSum(w*expr, compartment.content())
    elif len(compartments) == 2:
        i = iter(compartments.items())
        (compartment1, count1) = next(i)
        (compartment2, count2) = next(i)
        __checkSimpleCompartment(compartment1)
        __checkSimpleCompartment(compartment2)
        if count1 != 1 or count2 != 1:
            raise RuntimeError("Higher than 2nd order transitions are not implemented yet")
        w = n(compartment1) * (n(compartment2) - __kronecker(compartment1, compartment2))
        return CompartmentSum(CompartmentSum(w/2*expr, compartment1.content()), compartment2.content())
    else:
        raise RuntimeError("Higher than 2nd order transitions are not implemented yet")








# temporary export for playground

def checkSimpleCompartment(expr):
    __checkSimpleCompartment(expr)

def getMassAction(compartments):
    return __getMassAction(compartments)
