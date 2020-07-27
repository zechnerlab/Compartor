from sympy import *
import collections
import itertools
# from sympy import Function, Add, Mul, Integer, IndexedBase, factorial, ff, KroneckerDelta, Expr


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
def __getSumMassAction(compartments, expr):
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


# -------------------------------------------------
class ContentChange(Function):
    def __str__(self):
        return f'{self.args}'

    def _sympystr(self, printer=None):
        return f'{self.args}'

    def _latex(self, printer=None):
        return printer.doprint(self.args)


# -------------------------------------------------
def __getContentPerSpecies(content, D):
    """
    Get an array of scalars representing compartment content for species [0,D)

    For example,
        getContentPerSpecies(ContentVar('X') + ContentChange(0,-1,1), 3)
    returns
        [X[0], X[1] - 1, X[2] + 1]

    :param Expr content: the content of the compartment, comprising ContentVars, ContentChanges, sums of those, and multiplication by integers
    :param int D: the number of species
    :returns: list of scalar contents for species [0,D)
    """
    if content.func == Add:
        xs = [__getContentPerSpecies(arg, D) for arg in content.args]
        return [Add(*x) for x in zip(*xs)]
    elif content.func == Mul:
        xs = [__getContentPerSpecies(arg, D) for arg in content.args]
        return [Mul(*x) for x in zip(*xs)]
    elif content.func == IndexedBase:
        return [content[i] for i in range(D)]
    elif content.func == ContentChange:
        return [content.args[i] for i in range(D)]
    elif issubclass(content.func, Integer):
        return [content] * D
    else:
        raise TypeError("Unexpected expression " + str(content))




# temporary export for playground

def checkSimpleCompartment(expr):
    __checkSimpleCompartment(expr)

def getSumMassAction(compartments, expr=1):
    return __getSumMassAction(compartments, expr)

def getContentPerSpecies(content, D):
    return __getContentPerSpecies(content, D)
