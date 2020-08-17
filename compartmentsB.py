from compartmentsBase import Moment, Expectation
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
def __getSumMassAction(reactants, expr):
    """
    Get sum_Xc(w(n;Xc)*expr).

    :param dict reactants: reactant compartments Xc as a dictionary that maps Compartment to number of occurrences
    :param expr: expression in the sum
    :return: sum_Xc(w(n;Xc)*expr)
    """
    if len(reactants) == 0:
        return expr
    elif len(reactants) == 1:
        (compartment, count) = next(iter(reactants.items()))
        __checkSimpleCompartment(compartment)
        w = 1 / factorial(count) * ff(n(compartment), count)
        return CompartmentSum(w*expr, compartment.content())
    elif len(reactants) == 2:
        i = iter(reactants.items())
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


# -------------------------------------------------
def __mpow(content_per_species, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    """
    Get mul_(i=0..D)(x_i^gamma_i)

    :param content_per_species: list of compartment contents for species [0,D)
    :param Expr gamma: optional symbol to use for gamma
    :return: scalar expression for mul_(i=0..D)(x_i^gamma_i)
    """
    return Mul(*[content_per_species[i] ** gamma[i] for i in range(len(content_per_species))])


# -------------------------------------------------
def __deltaMContent(expr, D, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    """
    Compute delta M^gamma contribution for the given compartment content expr.

    :param Expr expr: the content of the compartment, comprising ContentVars, ContentChanges, sums of those, and multiplication by integers
    :param int D: the number of species
    :param Expr gamma: optional symbol to use for gamma
    :return:
    """
    if expr.func == Compartment:
        content = expr.args[0]
        species = __getContentPerSpecies(content, D)
        return __mpow(species, gamma)
    elif expr.func == EmptySet:
        return 0
    elif expr.func == Integer:
        return expr
    elif expr.func == Add:
        return Add(*[__deltaMContent(i) for i in expr.args])
    elif expr.func == Mul:
        return Mul(*[__deltaMContent(i) for i in expr.args])
    else:
        raise TypeError("Unexpected expression " + str(expr))


# -------------------------------------------------
def __deltaMCompartments(compartments, D, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    """
    Compute delta M^gamma term for the given compartments.

    Weights: Each lhs (Xc) occurrence counts -1. Each rhs (Yc) occurrence counts +1.

    :param dict compartments: reactant and product compartments as a dictionary that maps Compartment to weight.
    :param int D: the number of species
    :param Expr gamma: optional symbol to use for gamma
    :return:
    """
    if len(compartments) == 0:
        return 0
    else:
        return Add(*[__deltaMContent(cmp, D, gamma) * w for (cmp, w) in compartments.items()])


# -------------------------------------------------
def __deltaM(reactants, products, D, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    """

    :param dict reactants: reactant compartments Xc as a dictionary that maps Compartment to number of occurrences
    :param dict products: product compartments Yc as a dictionary that maps Compartment to number of occurrences
    :param int D: the number of species
    :param Expr gamma: optional symbol to use for gamma
    :return:
    """
    compartments = collections.defaultdict(int)
    for (compartment, count) in reactants.items():
        compartments[compartment] -= count
    for (compartment, count) in products.items():
        compartments[compartment] += count
    return __deltaMCompartments(compartments, D, gamma)


# -------------------------------------------------
def __substituteGamma(expr, *args, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    """
    Substitute gamma[i] by args[i] in expression.

    :param Expr expr: expression
    :param args: entries of the gamma vector
    :param Expr gamma: optional symbol to use for gamma
    :return: expr with gamma[i] substituted by args[i]
    """
    return expr.subs({gamma[i]: args[i] for i in range(len(args))})


# -------------------------------------------------

# temporary export for playground

def checkSimpleCompartment(expr):
    __checkSimpleCompartment(expr)

def getSumMassAction(compartments, expr=1):
    return __getSumMassAction(compartments, expr)

def getContentPerSpecies(content, D):
    return __getContentPerSpecies(content, D)

def mpow(content_per_species, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    return __mpow(content_per_species, gamma)

def deltaM(reactants, products, D, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    return __deltaM(reactants, products, D, gamma)

def substituteGamma(expr, *args, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    return __substituteGamma(expr, *args, gamma)



# -------------------------------------------------

# temporary functions for playground display

def yexp(reactants, products, pi_c, *gamma ):
    """
    TODO
    :param reactants:
    :param products:
    :param pi_c:
    :param gamma:
    :return:
    """
    # TODO:
    #  This doesn't compute any expectation or consider pi_c distribution yet
    yexp = deltaM(reactants, products, 1 ) * pi_c # TODO: 1 should be D!?
    yexp = substituteGamma(yexp, *gamma)
    return yexp

def lhs(*gamma):
    t=symbols('t')
    return Derivative(Expectation(Moment(*gamma)), t)

def rhs(reactants, products, k_c, g_c, pi_c, *gamma ):
    return k_c * Expectation(getSumMassAction(reactants, g_c * yexp(reactants, products, pi_c, *gamma)))
