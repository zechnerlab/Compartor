from sympy import Function, IndexedBase, Indexed, Basic, Symbol, EmptySet, \
    Add, Mul, Pow, Integer, Eq, KroneckerDelta, \
    factorial, ff, hessian, Matrix, reshape, symbols, simplify
from sympy.core.decorators import call_highest_priority

from time import time
from datetime import timedelta

import itertools
import collections

from sys import stderr

debugEnabled = False
infoEnabled = True
warningEnabled = True
def debug(*args, **kwargs):
    print(*args, file=stderr, **kwargs) if debugEnabled else None

def info(*args, **kwargs):
    print(*args, file=stderr, **kwargs) if infoEnabled else None

def warn(*args, **kwargs):
    print(*args, file=stderr, **kwargs) if warningEnabled else None

def error(*args, **kwargs):
    print(*args, file=stderr, **kwargs)
    raise RuntimeError("ERROR!")



###################################################
#
# Specifying transitions
#
###################################################

# -------------------------------------------------
class Content(IndexedBase):
    """
    A content variable.
    If X is a content variable, then X[i] refers to content for species i.
    """

    def __new__(cls, label, shape=None, **kw_args):
        return IndexedBase.__new__(cls, label, shape, **kw_args)

    @call_highest_priority('__radd__')
    def __add__(self, other):
        if type(other) is tuple:
            other = ContentChange(*other)
        elif type(other) is int:
            other = ContentChange(other)
        return Add(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        if type(other) is tuple:
            other = ContentChange(*other)
        elif type(other) is int:
            other = ContentChange(other)
        return Add(self, other)

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return self.__add__(-other)


# -------------------------------------------------
class ContentChange(Function):
    """
    An integer vector that can be added to a Content variable to express chemical modifications.
    args are change per species.
    """

    def __str__(self):
        return f'{self.args}'

    def _sympystr(self, printer=None):
        return f'{self.args}'

    def _latex(self, printer=None):
        return printer.doprint(self.args)


# -------------------------------------------------
class Compartment(Function):
    """
    Expression for a compartment, with one argument that is the expression for the compartment content.
    """
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
class Transition(Basic):
    """
    Expression for a transition with lhs and rhs specifying sums of compartments
    """

    def __new__(cls, lhs, rhs, name=None, isBulk=False):
        t = Basic.__new__(cls)
        t.lhs = lhs
        t.rhs = rhs
        t.name = name
        t.isBulk = isBulk
        return t

    def __str__(self):
        return f'{self.lhs} ---> {self.rhs}'

    def _sympystr(self, printer=None):
        return f'{self.lhs} ---> {self.rhs}'

    def _latex(self, printer=None, align=False, name=None):
        # Always use printer.doprint() otherwise nested expressions won't
        # work. See the example of ModOpWrong.
        l = printer.doprint(self.lhs)
        r = printer.doprint(self.rhs)
        if name is None:
            name = self.name
        arrow = '\longrightarrow{}' if name is None else '\overset{h_{' + name + '}}{\longrightarrow}'
        alignment = '&' if align else ''
        return l + alignment + arrow + r


###################################################
#
# Specifying propensities
#
###################################################

# -------------------------------------------------
def Constant(name):
    return Symbol(name, real=True, constant=True)


# -------------------------------------------------
class OutcomeDistribution(object):
    """
    Represents the probability distribution \pi_c() as
        - an expression `expr` to be used when displaying in equations (typically just a symbol \pi_c)
        - a function `conditional_expectation` that, given an expression computes its expectation (over Y_c)
    """

    def __init__(self, expr, conditional_expectation):
        self.expr = expr
        self.conditional_expectation = conditional_expectation

    def __repr__(self):
        return f'OutcomeDistribution({self.expr}, {self.conditional_expectation})'

    _identity = None

    @classmethod
    def Identity(cls):
        """
        Returns the OutcomeDistribution with identity conditional_expectation.
        This can be used for Y_c = {} or, more precisely,
        if the all content variables occurring in product compartments already occur in reactant compartments.
        """
        if not cls._identity:
            cls._identity = OutcomeDistribution(1, lambda x: x)
        return cls._identity

    @classmethod
    def CombineIndependent(cls, symbol, *distributions):
        """
        Combines an arbitrary number of previously defined outcome distributions.
        These should involve different chemical species each and be independent!
        """
        def expectation(pDMcj):
            cur = pDMcj.expand()
            # debug("> cur = %s" %(cur))
            for d in distributions:
                cur = d.conditional_expectation(cur)
                # debug("> cur = %s" %(cur))
            return cur

        return OutcomeDistribution(symbol, expectation)

    @classmethod
    def Constant(cls, symbol, y, cst):
        """
        Returns a constant distribution
        """
        def expectation(pDMcj):
            return pDMcj.subs(y, cst)

        return OutcomeDistribution(symbol, expectation)

    @classmethod
    def Poisson(cls, symbol, y, rate):
        """
        Returns an OutcomeDistribution that is a Poisson distribution of y

        :param symbol: symbol to use when displaying Pi_c in equations
        :param y: random variable, entry in a content variable, e.g., y[0]
        :param rate: lambda parameter of the Poisson distribution
        :return Pi_c:
        """
        # e.g.
        # y = y[0]
        # rate = Symbol("lambda", positive=True)
        from sympy.stats import Poisson, E
        def expectation(pDMcj):
            poiss = Poisson('poiss', rate)
            return E(pDMcj.subs(y, poiss))

        return OutcomeDistribution(symbol, expectation)

    @classmethod
    def NegativeBinomial(cls, symbol, y, r, p):
        """
        Returns an OutcomeDistribution that is a Negative Binomial distribution

        :param symbol: symbol to use when displaying Pi_c in equations
        :param y: random variable, entry in a content variable, e.g., y[0]
        :param r: failures parameter of the Negative Binomial distribution
        :param p: success probability of the Negative Binomial distribution.
        :return Pi_c:
        """
        from sympy.stats import NegativeBinomial, E
        def expectation(pDMcj):
            nb = NegativeBinomial('nb', r, p)
            return E(pDMcj.subs(y, nb))

        return OutcomeDistribution(symbol, expectation)

    @classmethod
    def Normal(cls, symbol, y, mu, sigma):
        """
        Returns an OutcomeDistribution that is a Normal distribution

        :param symbol: symbol to use when displaying Pi_c in equations
        :param y: random variable, entry in a content variable, e.g., y[0]
        :param mu: mean of the normal distribution
        :param sigma: sigma of the normal distribution.
        :return Pi_c:
        """
        from sympy.stats import Normal, E
        def expectation(pDMcj):
            n = Normal('n', mu, sigma)
            return E(pDMcj.subs(y, n))

        return OutcomeDistribution(symbol, expectation)

    @classmethod
    def Uniform(cls, symbol, y, start, end):
        """
        Returns an OutcomeDistribution that is a uniform distribution of y with values from start (inclusive) to end (inclusive)

        :param symbol: symbol to use when displaying Pi_c in equations
        :param y: random variable, entry in a content variable, e.g., y[0]
        :return Pi_c:
        """
        # e.g.
        # y = y[0]
        # start = 0
        # end = x[0]
        from sympy import Sum
        def expectation(pDMcj):
            p = end - start + 1
            exp = Sum(
                pDMcj,
                (y, start, end)
            ).doit()
            exp /= p
            return exp.factor()

        return OutcomeDistribution(symbol, expectation)

    @classmethod
    def StemCell(cls, symbol, D, y, yp, x, xp):
        """
        Returns an OutcomeDistribution that matches the StemCell case study of Duso2020.

        :param symbol: symbol to use when displaying Pi_c in equations
        :param y: random variable, entry in a content variable, e.g., y
        :param yp: random variable, entry in a content variable, e.g., yp
        :param x: random variable, entry in a content variable, e.g., x
        :param xp: random variable, entry in a content variable, e.g., xp
        :return Pi_c:
        """
        import sys

        def expectation(pDMcj):
            a = pDMcj.subs( y[0], 1).subs( y[1], x[1])
            a =     a.subs(yp[0], 0).subs(yp[1], 0)
            for i in range(2,D):
                a = a.subs( y[i], x[i]).subs(yp[i],xp[i])
            return a

        return OutcomeDistribution(symbol, expectation)

    @classmethod
    def Binomial(cls, symbol, y, n, p):
        """
        Returns an OutcomeDistribution that is a Binomial distribution

        :param symbol: symbol to use when displaying Pi_c in equations
        :param y: random variable, entry in a content variable, e.g., y[0]
        :param n: number parameter of the Binomial distribution
        :param p: success probability of the Binomial distribution.
        :return Pi_c:
        """
        from sympy.stats import Binomial, E
        from sympy import Sum
        def expectation(pDMcj):
            binomial = Binomial('binomial', n, p)
            exp = E(pDMcj.subs(y, binomial))
            # def _bin(n,k):
            #     return factorial(n) / ( factorial(k)*factorial(n-k) )
            # exp = Sum(
            #             pDMcj * p**y * (1-p)**(n-y) * _bin(n,y),
            #             (y, 0, n)
            #             ).doit()
            # exp /= n+1
            # debug(">> Binomial: exp = %s" %(exp)) #debug
            return exp

        return OutcomeDistribution(symbol, expectation)


###################################################
#
# Specifying transitions classes
#
###################################################

# -------------------------------------------------
class TransitionClass(Basic):
    """
    Transition class comprising a Transition, a content-independent rate constant k, 
    a reactant tuning function g, and the outcome distribuiton pi
    """

    def __new__(cls, transition, k, g=1, pi=OutcomeDistribution.Identity(), name=None):
        t = Basic.__new__(cls)
        if pi == OutcomeDistribution.Identity():
            cvl = _getContentVars(transition.lhs)
            cvr = _getContentVars(transition.rhs)
            if cvr - cvl:
                raise ValueError("Please specify an OutcomeDistribution!"
                                 " Content variables occur in products that do not occur in reactants."
                                 " The default OutcomeDistribution cannot be applied.")

        if name is None:
            name = transition.name

        if isinstance(k, str):
            k = Constant(k)

        t.transition = transition
        t.name = name
        t.k = k
        t.g = g
        t.pi = pi
        return t

    def __str__(self):
        return f'TransitionClass("{self.name}", {self.transition}, k={self.k}, g={self.g}, pi={self.pi})'

    def _propensity_str(self, name=None):
        if name is None:
            name = self.transition.name
        if name is None:
            name = ''
        w = (self.transition.lhs)
        if not self.transition.isBulk:
            reactants = getCompartments(self.transition.lhs)
            w = _getWnXc(reactants)
        expr = self.k * self.g * self.pi.expr * w
        return "h_" + name + " = " + str(expr)

    def _sympystr(self, printer=None):
        t = printer.doprint(self.transition)
        p = self._propensity_latex(printer)
        return r"(%s, %s)" % (t,p)

    def _latex(self, printer=None):
        transition_latex = self.transition._latex(printer, name=self.name)
        propensity_latex = self._propensity_latex(printer, name=self.name)
        return r"%s,\:%s" % (transition_latex, propensity_latex)

    def _propensity_latex(self, printer=None, name=None):
        if name is None:
            name = self.transition.name
        if name is None:
            name = ''
        h_c = Symbol("h_{" + name + "}")
        w = (self.transition.lhs)
        if not self.transition.isBulk:
            reactants = getCompartments(self.transition.lhs)
            w = _getWnXc(reactants)
        expr = self.k * self.g * self.pi.expr * w
        return printer.doprint(Eq(h_c, expr, evaluate=False))


# -------------------------------------------------
def _getContentVars(expr):
    """
    Get all the Content variables occurring in expr.
    (This is used in TransitionClass to check whether the default OutcomeDistribution.Identity() is permissible)

    :param Expr expr: Compartment, Content, ContentChange, sums of those, and multiplication by integers
    :returns: set of Content variables
    """
    import sys

    if expr.func in [Add, Mul, Pow, Compartment, ContentChange] \
            or issubclass(expr.func, Function):
        return set(itertools.chain(*(_getContentVars(arg) for arg in expr.args)))
    elif expr.func == Content:
        return {expr}
    elif expr.func == Indexed:
        return {expr.base}
    elif expr is EmptySet:
        return set()
    elif issubclass(expr.func, Integer):
        return set()
    elif len(expr.args) == 0:
        return set()
    else:
        warn("Warning: Unexpected expression " + str(expr) + ", taking default branch!")
        return set(itertools.chain(*(_getContentVars(arg) for arg in expr.args)))
        # raise TypeError("Unexpected expression " + str(expr))


# -------------------------------------------------
def _getWnXc(reactants):
    """
    Get w(n;Xc).
    (This is used for displaying propensities only)

    :param dict reactants: reactant compartments Xc as a dictionary that maps Compartment to number of occurrences
    :return: w(n;Xc)
    """

    def _n(content):
        """
        Expression for number of compartments with given content
        """
        if content.func == Compartment:
            return _n(content.args[0])
        return Function('n', integer=True)(content)

    def _kronecker(content1, content2):
        if content1.func == Compartment:
            return _kronecker(content1.args[0], content2)
        if content2.func == Compartment:
            return _kronecker(content1, content2.args[0])
        return KroneckerDelta(content1, content2)

    if len(reactants) == 0:
        return 1
    elif len(reactants) == 1:
        (compartment, count) = next(iter(reactants.items()))
        __checkSimpleCompartment(compartment)
        return 1 / factorial(count) * ff(_n(compartment), count)
    elif len(reactants) == 2:
        i = iter(reactants.items())
        (compartment1, count1) = next(i)
        (compartment2, count2) = next(i)
        __checkSimpleCompartment(compartment1)
        __checkSimpleCompartment(compartment2)
        if count1 != 1 or count2 != 1:
            raise RuntimeError("Higher than 2nd order transitions are not implemented yet")
        return _n(compartment1) * (_n(compartment2) - _kronecker(compartment1, compartment2)) \
               / (1 + _kronecker(compartment1, compartment2))
    else:
        raise RuntimeError("Higher than 2nd order transitions are not implemented yet")


###################################################
#
# Moment symbol, DeltaM symbol
#
###################################################

# -------------------------------------------------
class Moment(Function):
    """
    Expression for M^\gamma, args are the elements of \gamma
    """

    def __str__(self):
        return f'Moment{self.args}'

    def _latex(self, printer=None, exp=1):
        b = self.__base_latex(printer=printer)
        if exp == 1:
            return b
        else:
            return '{\\left(' + b + '\\right)}^{' + printer.doprint(exp) + '}'

    def __base_latex(self, printer=None):
        if len(self.args) == 0:
            return 'M^{\gamma}'
        elif len(self.args) == 1:
            return 'M^{' + printer.doprint(self.args[0]) + '}'
        else:
            return 'M^{\\left(' + ", ".join([printer.doprint(arg) for arg in self.args]) + '\\right)}'

    def order(self):
        return sum(self.args)


# -------------------------------------------------
class DeltaM(Function):
    """
    Expression for \Delta{}M^\gamma, args are the elements of \gamma
    """

    def __str__(self):
        return f'DeltaM{self.args}'

    def _latex(self, printer=None, exp=1):
        b = self.__base_latex(printer=printer)
        if exp == 1:
            return b
        else:
            return '{\\left(' + b + '\\right)}^{' + printer.doprint(exp) + '}'

    def __base_latex(self, printer=None):
        if len(self.args) == 0:
            return '\Delta{}M^{\gamma}'
        elif len(self.args) == 1:
            return '\Delta{}M^{' + printer.doprint(self.args[0]) + '}'
        else:
            return '\Delta{}M^{\\left(' + ", ".join([printer.doprint(arg) for arg in self.args]) + '\\right)}'

### Test Bulk stuff
class Bulk(Function):
    """
    Expression for B^\gamma, args are the elements of \gamma
    """

    def __str__(self):
        return f'Bulk{self.args}'

    def _latex(self, printer=None, exp=1):
        b = self.__base_latex(printer=printer)
        if exp == 1:
            return b
        else:
            return '{\\left(' + b + '\\right)}^{' + printer.doprint(exp) + '}'

    def __base_latex(self, printer=None):
        if len(self.args) == 0:
            return 'B^{\gamma}'
        elif len(self.args) == 1:
            return 'B^{' + printer.doprint(self.args[0]) + '}'
        else:
            return 'B^{\\left(' + ", ".join([printer.doprint(arg) for arg in self.args]) + '\\right)}'

    def order(self):
        return sum(self.args)

###################################################
#
# Expectation function
#
###################################################

# -------------------------------------------------
class Expectation(Function):
    """
    just used for displaying <...>
    """
    nargs = 1

    def __str__(self):
        return f'E[{self.args[0]}]'

    def _sympystr(self, printer=None):
        return f'E[{self.args[0]}]'

    def _latex(self, printer=None, exp=1):
        b = self.__base_latex(printer=printer)
        if exp == 1:
            return b
        else:
            return '{' + b + '}^{' + printer.doprint(exp) + '}'

    def __base_latex(self, printer=None):
        return '\\left< ' + printer.doprint(self.args[0]) + '\\right> '


###################################################
#
# Derivative of expression in Moments using Ito's rule
#
###################################################

# -------------------------------------------------
def __getMoments(expr):
    """
    Get all instances of Moment(...) occurring in expr
    :param expr:
    :return:
    """
    if expr.func == Add or expr.func == Mul:
        moments = [__getMoments(arg) for arg in expr.args]
        return list(set(itertools.chain(*moments)))
    elif expr.func == Pow:
        return __getMoments(expr.args[0])
    elif expr.func == Moment:
        return [expr]
    elif issubclass(expr.func, Integer):
        return []
    else:
        raise TypeError("Unexpected expression " + str(expr))


# -------------------------------------------------
def ito(expr):
    """
    Get derivative of a function of moments using Ito's rule
    :param expr: expression comprising Moments, addition, multiplication, and powers with moments only in the base
    :return: derivative obtained by Ito's rule
    """
    moments = __getMoments(expr)
    substitutions = [(m, m + DeltaM(*m.args)) for m in moments]
    expr = expr.subs(substitutions) - expr
    return expr.expand()


###################################################
#
# Computing df(M)/dt
#
###################################################

# -------------------------------------------------
def getCompartments(expr):
    """
    Extract a dictionary that maps Compartment to number of occurrences from a compartment expression.

    :param expr: sum of integer multiples of Compartments. (Typically lhs or rhs of a Transition.)
    :return: expr as a dictionary that maps Compartment to number of occurrences
    """
    if expr.func == Add:
        summands = [*expr.args]
    else:
        summands = [expr]

    compartments = collections.defaultdict(int)
    for expr in summands:
        if expr.func == Mul and expr.args[0].func == Integer and expr.args[1].func == Compartment:
            count = expr.args[0]
            compartment = expr.args[1]
        elif expr.func == Compartment:
            count = 1
            compartment = expr
        elif expr == EmptySet:
            continue
        else:
            raise TypeError("Unexpected expression " + str(expr))
        compartments[compartment] += count

    return compartments


# -------------------------------------------------
def decomposeMomentsPolynomial(expr, strict=True):
    """
    Split a polynomial in M^{\gamma^k} and \DeltaM^{\gamma^l} into a list of monomials.

    :param expr: a polynomial in M^{\gamma^k} and \DeltaM^{\gamma^l}.
    :param strict: if True, only allow integers in constant of each monomial.
            If False, everything that is not a Moment or DeltaMoment counts as constant.
    :return: list of monomials, each decomposed as a tuple (constant, product of Moments, product of DeltaMoments)
    """
    expr = expr.expand()
    monomials = list(expr.args) if expr.func == Add else [expr]
    result = list()
    for monomial in monomials:
        factors = list(monomial.args) if monomial.func == Mul else [monomial]
        qK = 1
        qM = 1
        qDM = 1
        for factor in factors:
            if factor.func == Moment:
                qM *= factor
            elif factor.func == DeltaM:
                qDM *= factor
            elif issubclass(factor.func, Integer):
                qK *= factor
            elif factor.func == Pow and issubclass(factor.args[1].func, Integer):
                if factor.args[0].func == Moment:
                    qM *= factor
                elif factor.args[0].func == DeltaM:
                    qDM *= factor
                elif strict is False:
                    qK *= factor
                else:
                    raise TypeError("Unexpected expression " + str(factor))
            elif strict is False:
                qK *= factor
            else:
                raise TypeError("Unexpected expression " + str(factor))
        result.append((qK, qM, qDM))
    return result


# -------------------------------------------------
def __getContentPerSpecies(content, D):
    """
    Get an array of scalars representing compartment content for species [0,D)

    For example,
        getContentPerSpecies(Content('X') + ContentChange(0,-1,1), 3)
    returns
        [X[0], X[1] - 1, X[2] + 1]

    :param Expr content: the content of the compartment, comprising Contents, ContentChanges, sums of those, and multiplication by integers
    :param int D: the number of species
    :returns: list of scalar contents for species [0,D)
    """
    if content.func == Add:
        xs = [__getContentPerSpecies(arg, D) for arg in content.args]
        return [Add(*x) for x in zip(*xs)]
    elif content.func == Mul:
        xs = [__getContentPerSpecies(arg, D) for arg in content.args]
        return [Mul(*x) for x in zip(*xs)]
    elif content.func == Content:
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

    :param Expr expr: the content of the compartment, comprising Contents, ContentChanges, sums of those, and multiplication by integers
    :param int D: the number of species
    :param Expr gamma: optional symbol to use for gamma
    :return:
    """
    if expr.func == Compartment:
        content = expr.args[0]
        species = __getContentPerSpecies(content, D)
        return __mpow(species, gamma)
    elif expr == EmptySet:
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
    Compute \DeltaM^\gamma term for the given compartments.

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
def getDeltaM(reactants, products, D, gamma=IndexedBase('\gamma', integer=True, shape=1)):
    """
    Derive \DeltaM_{c,j}^\gamma for the given reactant and product compartments (of transition c).
    The returned expression does not yet have \gamma instantiated by concrete integers.

    :param dict reactants: reactant compartments Xc as a dictionary that maps Compartment to number of occurrences
    :param dict products: product compartments Yc as a dictionary that maps Compartment to number of occurrences
    :param int D: the number of species
    :param Expr gamma: optional symbol to use for gamma
    :return: expression of \DeltaM_{c,j}^\gamma in terms of content variable entries.
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
def subsDeltaM(expr, deltaM):
    """
    Replace every DeltaM(g) symbol in expr by deltaM with \gamma substituted with g.

    :param expr: expression containing DeltaM symbols
    :param deltaM: expression to substitute for DeltaM (with uninstantiated \gamma)
    :return: expr with every DeltaM(g) symbol replaced.
    """
    if expr.func == DeltaM:
        return __substituteGamma(deltaM, *expr.args)
    elif expr.func == Pow:
        return Pow(subsDeltaM(expr.args[0], deltaM), expr.args[1])
    elif expr.func == Add:
        return Add(*[subsDeltaM(arg, deltaM) for arg in expr.args])
    elif expr.func == Mul:
        return Mul(*[subsDeltaM(arg, deltaM) for arg in expr.args])
    elif issubclass(expr.func, Integer):
        return expr
    else:
        raise TypeError("Unexpected expression " + str(expr))


# -------------------------------------------------
def __getKernel(moment, symbol):
    assert moment.func == Moment
    alpha = moment.args
    kernel = 1
    for i,j in enumerate(alpha):
        kernel *= symbol[i]**j
    return kernel

def __getKnownMomentSubstitution(moment):
    _x = Content('_x')
    k = __getKernel(moment, _x)

def __getKnownMomentsSubstitutions(known_moments):
    blabla

# -------------------------------------------------
def __extractMonomials(expr):
    # expr = expr.factor().expand() # Slow factor()
    expr = expr.expand() # Faster, and it should be same result
    monomials = list(expr.args) if expr.func == Add else [expr]
    return monomials

def __containsMoments(expr):
    func = expr.func
    args = expr.args
    if func == Moment:
        return True
    elif len(args) == 0:
        return False
    else:
        lower = [ __containsMoments(e) for e in args ]
        return any(lower)

def __containsSymbol(expr, s, D):
    return any( [ s[i] in expr.free_symbols for i in range(D) ] )

def __takeExpectationOfMoments(expr):
    expr = expr.factor().expand()
    args = list(expr.args) if expr.func == Mul else [expr]
    argsExp = [ Expectation(x) if type(x) == Moment else x for x in args ]
    return Mul(*argsExp)

def __extractMoment1(monomial, x, D):
    factors = list(monomial.args) if monomial.func == Mul else [monomial]
    k = 1
    alpha = [0] * D
    for factor in factors:
        if factor.func == Pow \
                and factor.args[0].func == Indexed \
                and factor.args[0].args[0] == x \
                and issubclass(factor.args[1].func, Integer):
            alpha[factor.args[0].args[1]] = factor.args[1]
        elif factor.func == Indexed and factor.args[0] == x:
            alpha[factor.args[1]] = 1
        else:
            k *= factor
    return (k, alpha)

def __extractMoment2(monomial, x, y, D):
    factors = list(monomial.args) if monomial.func == Mul else [monomial]
    k = 1
    alpha = [0] * D
    beta = [0] * D
    for factor in factors:
        if factor.func == Pow \
                and factor.args[0].func == Indexed \
                and issubclass(factor.args[1].func, Integer):
            cvar = factor.args[0].args[0]
            cidx = factor.args[0].args[1]
            if cvar == x:
                alpha[cidx] = factor.args[1]
                continue
            elif cvar == y:
                beta[cidx] = factor.args[1]
                continue
        elif factor.func == Indexed:
            cvar = factor.args[0]
            cidx = factor.args[1]
            if cvar == x:
                alpha[cidx] = 1
                continue
            elif cvar == y:
                beta[cidx] = 1
                continue
        k *= factor
    return (k, alpha, beta)

def getMomentsInExpression(expr):
    monomials = decomposeMomentsPolynomial(expr, strict=False)
    # debug("getMomentsInExpression: expr=%s, monomials=%s" %(expr, monomials))
    moments = set()
    for (k, M, DM) in monomials:
        if M != 1:
            debug("getMomentsInExpression: adding M=%s" %(M))
            moments.add(M)
    return moments

def __isPowOfMoment(expr):
    return expr.func == Pow and expr.args[0].func == Moment

def __isMomentOrPowOfMoment(expr):
    return expr.func == Moment or __isPowOfMoment(expr)

def __extractExpectedMomentsFromTerm(expr):
    S = set()
    if expr == 1:
        pass
    elif expr.func == Expectation:
        A = expr.args[0]
        if __isMomentOrPowOfMoment(A):
            S.add(A)
        elif A.func == Mul and all( [ __isMomentOrPowOfMoment(M) for M in A.args] ):
            S.add(A)
        else:
            warn(">>> WRN: Weird Expected stuff: A=%s, A.func=%s, A.args=%s" %(A, A.func, A.args))
    elif expr.func == Mul:
        for T in expr.args:
            S.update( __extractExpectedMomentsFromTerm(T) )
    elif expr.func == Pow:
        S.update( __extractExpectedMomentsFromTerm(expr.args[0]) )
    else:
        pass
    return S

def getExpectedMomentsInExpression(expr):
    debug("getExpectedMomentsInExpression: expr=%s" %(expr))
    monomials = decomposeMomentsPolynomial(expr, strict=False)
    Emoments = set()
    for (k, M, DM) in monomials:
        debug("getExpectedMomentsInExpression: k=%s, M=%s, DM=%s" %(k,M,DM))
        S = __extractExpectedMomentsFromTerm(k)
        S.update( __extractExpectedMomentsFromTerm(DM) )
        if len(S) > 0:
            debug("getExpectedMomentsInExpression: adding S=%s" %(S))
            Emoments.update(S)
    return Emoments

def __decomposeMomentMonomial(expr):
    S = set()
    func = expr.func
    if func == Moment:
        S.add(expr)
    elif func == Pow:
        S.add(expr.args[0])
    elif func == Mul:
        for M in expr.args:
            S.update( __decomposeMomentMonomial(M) )
    else:
        error("FATAL: unknown function found in moments monomials")
    return S

def getSingleMomentsInExpression(expr):
    moments = getMomentsInExpression(expr)
    sMoments = set()
    for M in moments:
        sMoments.update( __decomposeMomentMonomial(M) )
    return sMoments

# -------------------------------------------------
def __decomposeContentPolynomial(expr, x, D, order=2, boolean_variables=set(), lpac=False, lpPower=1):
    """
    Given a polynomial in Xc = {x}, decompose its monomials as (constant * prod x[i]^alpha[i])

    :param expr: a polynomial in Xc = {x}.
    :param x: content variable x
    :param D: number of species
    :return: list of monomials, each decomposed into (constant, alpha)
    """
    if lpac:
        expr = _expandContentFunction(expr, D, order=order, boolean_variables=boolean_variables, lpPower=lpPower)
        # expr = expr.factor()
    monomials = __extractMonomials(expr)
    result = list()
    for monomial in monomials:
        k, alpha = __extractMoment1(monomial, x, D)
        result.append((k, alpha))
    return result


# -------------------------------------------------
def __decomposeContentPolynomial2(expr, x, y, D, order=2, boolean_variables=set(), lpac=False, lpPower=1):
    """
    Given a polynomial in Xc = {x, y}, decompose its monomials as (constant * prod x[i]^alpha[i] * prod y[i]^beta[i])

    :param expr: a polynomial in Xc = {x,y}.
    :param x: content variable x
    :param y: content variable y
    :param D: number of species
    :return: list of monomials, each decomposed into (constant, alpha, beta)
    """
    
    if lpac:
        expr = _expandContentFunction(expr, D, order=2, boolean_variables=boolean_variables, lpPower=lpPower)
        # expr = expr.factor()
    monomials = __extractMonomials(expr)
    result = list()
    for monomial in monomials:
        k, alpha, beta = __extractMoment2(monomial, x, y, D)
        result.append((k, alpha, beta))
    return result


# -------------------------------------------------
def __checkSimpleCompartment(expr):
    """
    Checks that expr is a Compartment(Content) and throws TypeError if it is not
    """
    if not (expr.func == Compartment and len(expr.args) == 1 and expr.args[0].func == Content):
        raise TypeError(
            "Only compartments comprising a single content variable are supported (not '" + str(expr) + "')")


# -------------------------------------------------
def _getNumSpecies(expr):
    """
    Extract number of species from a moments expression.
    Raise RuntimeError, if no Moment occurs in expr, or if occurring Moments have mismatching arities.
    """
    debug("_getNumSpecies: expr=%s" %(expr)) #debug
    if expr.func == Bulk:
        return len(expr.args)

    monomials = decomposeMomentsPolynomial(expr, strict=False)
    Ds = set()
    for (k, M, DM) in monomials:
        if M != 1:
            factors = list(M.args) if M.func == Mul else [M]
            for factor in factors:
                if factor.func == Pow and factor.args[0].func == Moment:
                    Ds.add(len(factor.args[0].args))
                elif factor.func == Moment:
                    Ds.add(len(factor.args))
    if len(Ds) == 0:
        raise RuntimeError("Cannot determine number of species from expression: " + str(expr))
    elif len(Ds) == 1:
        return next(iter(Ds))
    else:
        raise RuntimeError("Number of species in Moments occurring in expression is not unique. "
                           + str(expr) + " contains moments of orders " + str(Ds))

def _sanitizeGammaForBooleans(gamma, boolean_variables):
    for i in boolean_variables:
        if gamma[i] > 1:
            gamma[i] = 1 # Higher powers don't matter here!
    return gamma

def _sanitizeMonomialsForBooleans(monomials, boolean_variables):
    for (i, mon) in enumerate(monomials):
        if len(mon) == 2:
            k, alpha = mon
            monomials[i] = ( k, _sanitizeGammaForBooleans(alpha, boolean_variables) )
        elif len(mon) == 3:
            k, alpha, beta = mon
            monomials[i] = ( 
                        k, 
                        _sanitizeGammaForBooleans(alpha, boolean_variables),
                        _sanitizeGammaForBooleans(beta, boolean_variables) 
                        )
        else:
            raise RuntimeError("Monomials must have either 2 or 3 elements! Got: %d" %(len(mon)))
    return monomials

def _areBooleansInvolved(alpha, beta, boolean_variables):
    # print("CHECK BOOLS: alpha=%s, beta=%s, boolean_variables=%s"%(alpha, beta, boolean_variables), file=stderr) #debug
    for i in boolean_variables:
        if alpha[i]>0 or beta[i]>0:
            error("BOOLEANS INVOLVED! alpha=%s, beta=%s"%(alpha, beta)) #debug
            return True
    return False

# -------------------------------------------------
def _getGradient(cf, D, Vars):
    return [ cf.diff(v[i]) for i in range(D) for v in Vars ]
    # return [ cf.diff(v[i]).subs(KroneckerDelta(1,v[1]),1).simplify() for i in range(D) for v in Vars ]

def _evaluateMultidim(expr, D, Vars, linearizationPoint):
    e = expr
    for i in range(D):
        for v in Vars:
            if linearizationPoint[i] != None:
                e = e.subs(v[i], linearizationPoint[i])
    return e

def _getExpansionPoint(D, boolean_variables, order=1):
    E, M = Expectation, Moment
    lp = []
    zeroth = (0,)*D
    N = M(*zeroth)
    for i in range(D):
        linearizationPoint = None
        if i not in boolean_variables:
            exponent = ( *( (0,)*i ), order, *( (0,)*(D-i-1) ) )
            Mi = M(*exponent)
            linearizationPoint = E(Mi) / E(N)
            # Test approximation on linearizing fractional moments
            # linearizationPoint = Mi / N #Ehm...
            # linearizationPoint += E(Mi)*E(N**2) / E(N)**3
            # linearizationPoint -= E(Mi*N) / E(N)**2
        lp.append(linearizationPoint)
    return lp

def _linearizeContentFunction(cf, D, boolean_variables=set(), ignored_symbols=set(), order=1):
    # print("> cf = %s" %(cf), file=stderr) #debug
    Vars = _getContentVars(cf).difference(ignored_symbols)
    if len(Vars)==0:
        return cf
    # lp = _getExpansionPoint(D, boolean_variables) # Exclude booleans from lin
    lp = _getExpansionPoint(D, set(), order=order) # Let's linearize everything -> This works!

    Grad = _getGradient(cf, D, Vars)
    cf0 = _evaluateMultidim(cf, D, Vars, lp)
    Grad0 = [ _evaluateMultidim(g, D, Vars, lp) for g in Grad ]
    lcf = cf0
    j=0
    for i in range(D):
        # Iterate on all the species
        for v in Vars:
            # Iterate on all the content-symbols (if species not boolean)
            if i not in boolean_variables:
                lcf += Grad0[j] * (v[i] - lp[i])
            j += 1

    return lcf

def _multivariateTaylor(function_expression, variable_list, evaluation_point, degree):
    """
    Ref: https://stackoverflow.com/a/63850672

    Mathematical formulation reference:
    https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Multivariable_Calculus/3%3A_Topics_in_Partial_Derivatives/Taylor__Polynomials_of_Functions_of_Two_Variables
    :param function_expression: Sympy expression of the function
    :param variable_list: list. All variables to be approximated (to be "Taylorized")
    :param evaluation_point: list. Coordinates, where the function will be expressed
    :param degree: int. Total degree of the Taylor polynomial
    :return: Returns a Sympy expression of the Taylor series up to a given degree, of a given multivariate expression, approximated as a multivariate polynomial evaluated at the evaluation_point
    """
    from sympy import factorial, Matrix, prod
    import itertools

    n_var = len(variable_list)
    point_coordinates = [(i, j) for i, j in (zip(variable_list, evaluation_point))]  # list of tuples with variables and their evaluation_point coordinates, to later perform substitution

    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))  # list with exponentials of the partial derivatives
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]  # Discarding some higher-order terms
    n_terms = len(deriv_orders)
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]  # Individual degree of each partial derivative, of each term

    polynomial = 0
    for i in range(n_terms):
        partial_derivatives_at_point = function_expression.diff(*deriv_orders_as_input[i]).subs(point_coordinates)  # e.g. df/(dx*dy**2)
        denominator = prod([factorial(j) for j in deriv_orders[i]])  # e.g. (1! * 2!)
        distances_powered = prod([(Matrix(variable_list) - Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  # e.g. (x-x0)*(y-y0)**2
        polynomial += partial_derivatives_at_point / denominator * distances_powered
    return polynomial

def _expandContentFunction(cf, D, order=2, boolean_variables=set(), ignored_symbols=set(), lpPower=1):
    Vars = _getContentVars(cf).difference(ignored_symbols)
    if len(Vars)==0:
        return cf
    ep = _getExpansionPoint(D, set(), order=lpPower) # Let's linearize everything -> This works!

    varList = [ v[i] for i in range(D) for v in Vars ]
    evalPoints = [ ep[i] for i in range(D) for v in Vars ]
    ecf = _multivariateTaylor(cf, varList, evalPoints, order)

    return ecf

def _getBooleanVariablesIndices(expr, D):
    # Check if the current expression contains any KroneckerDelta() and keep
    # track of the indices where they appear!
    Vars = _getContentVars(expr)
    BV = set()
    for v in Vars:
        for i in range(D):
            check = expr.find( KroneckerDelta(v[i], 1) )
            check.update( expr.find( KroneckerDelta(v[i], 0) ) )
            if len(check)>0:
                BV.add(i)
    return BV

def _handleBooleanVariables(expr, D, boolean_variables):
    Vars = _getContentVars(expr)

    for (j,v) in enumerate(Vars):
        for i in boolean_variables:
            # Dummy temporary symbols for KroneckerDeltas
            k0, k1 = symbols('k0 k1')
            # Folding powers 1:3 into one symbol
            sub1 = [ ( KroneckerDelta(1, v[i])**k, k1 ) for k in range(1,4) ]
            sub0 = [ ( KroneckerDelta(0, v[i])**k, k0 ) for k in range(1,4) ]
            expr = expr.subs(sub1)
            expr = expr.subs(sub0)
            # Make sure that KroneckerDelta(x,0)*x -> 0
            expr = expr.subs(k0*v[i], 0)
            # Now remove any power of v[i] by substituting it with 1
            expr = expr.subs(v[i], 1)
            # Now replace the KroneckerDeltas' dummies with their "meaning"
            expr = expr.subs(k1, v[i])
            expr = expr.subs(k0, 1-v[i] )
    return expr

def _getSpeciesIndicesFromGamma(gamma):
    # Get the gamma (exponent) of a moment and extract the indices of the involved variables
    return [ i for i,e in enumerate(gamma) if e>0 ]

# -------------------------------------------------
def get_dfMdt_contrib(reactants, l_n_Xc, D, order=2, boolean_variables=set(), lpPower=1):
    """
    Compute the contribution to df(M)/dt of a particular transition and a particular monomial.

    :param reactants:
    :param l_n_Xc:
    :param D:
    :return:
    """
    if len(reactants) == 0:
        # In this case Xc={}
        return l_n_Xc
    elif len(reactants) == 1:
        (compartment, count) = next(iter(reactants.items()))
        __checkSimpleCompartment(compartment)
        if count != 1:
            raise RuntimeError("not implemented yet")
        # In this case Xc={x}
        # compartment=[x]
        x = compartment.args[0]
        monomials = __decomposeContentPolynomial(l_n_Xc, x, D, order=order, lpac=lpac, lpPower=lpPower)
        monomials = _sanitizeMonomialsForBooleans(monomials, boolean_variables)
        replaced = [k * Moment(*alpha) for (k, alpha) in monomials]
        return Add(*replaced)
    elif len(reactants) == 2:
        i = iter(reactants.items())
        (compartment1, count1) = next(i)
        (compartment2, count2) = next(i)
        __checkSimpleCompartment(compartment1)
        __checkSimpleCompartment(compartment2)
        if count1 != 1 or count2 != 1:
            raise RuntimeError("Higher than 2nd order transitions are not implemented yet")
        # In this case Xc={x, x'}
        # compartment1=[x]
        # compartment2=[x']
        x = compartment1.args[0]
        x1 = compartment2.args[0]
        monomials = __decomposeContentPolynomial2(l_n_Xc, x, x1, D, order=order, lpac=lpac, lpPower=lpPower)
        monomials = _sanitizeMonomialsForBooleans(monomials, boolean_variables)
        replaced1 = [k / 2 * Moment(*alpha) * Moment(*beta) for (k, alpha, beta) in monomials]
        monomials = __decomposeContentPolynomial(l_n_Xc.subs(x1, x), x, D, order=order, lpac=lpac, lpPower=lpPower)
        monomials = _sanitizeMonomialsForBooleans(monomials, boolean_variables)
        replaced2 = [k / 2 * Moment(*alpha) for (k, alpha) in monomials]
        return Add(*replaced1) - Add(*replaced2)
    else:
        raise RuntimeError("Higher than 2nd order transitions are not implemented yet")

# -------------------------------------------------
def get_dfMdt_contrib_lpac(reactants, contextFun, contentFun, D, 
                            order=2,
                            boolean_variables=set(),
                            lpPower=1,
                            ):
    """
    Compute the contribution to df(M)/dt of a particular transition and a particular monomial using the LPAC approximation.

    :param reactants:
    :param contextFun:
    :param contentFun:
    :param D:
    :return:
    """
    if type(reactants) in [Moment] or len(reactants) == 0:
        return contextFun * contentFun
    elif len(reactants) == 1:
        (compartment, count) = next(iter(reactants.items()))
        __checkSimpleCompartment(compartment)
        if count != 1:
            raise RuntimeError("not implemented yet")
        x = compartment.args[0]
        monomials = __decomposeContentPolynomial(contextFun * contentFun, x, D, 
                                                    order=order,
                                                    boolean_variables=boolean_variables,
                                                    lpac=True,
                                                    lpPower=lpPower,
                                                    )
        monomials = _sanitizeMonomialsForBooleans(monomials, boolean_variables)
        replaced = [k * Moment(*alpha) for (k, alpha) in monomials]
        return Add(*replaced)
    elif len(reactants) == 2:
        i = iter(reactants.items())
        (compartment1, count1) = next(i)
        (compartment2, count2) = next(i)
        __checkSimpleCompartment(compartment1)
        __checkSimpleCompartment(compartment2)
        if count1 != 1 or count2 != 1:
            raise RuntimeError("Higher than 2nd order transitions are not implemented yet")
        x = compartment1.args[0]
        x1 = compartment2.args[0]
        monomials = __decomposeContentPolynomial2(contextFun * contentFun, x, x1, D,
                                                    order=order, boolean_variables=boolean_variables,
                                                    lpac=True,
                                                    lpPower=lpPower,
                                                    )
        monomials = _sanitizeMonomialsForBooleans(monomials, boolean_variables)
        replaced1 = [k / 2 * Moment(*alpha) * Moment(*beta) for (k, alpha, beta) in monomials]
        monomials = __decomposeContentPolynomial((contextFun * contentFun).subs(x1, x), x, D,
                                                    order=order, boolean_variables=boolean_variables,
                                                    lpac=True,
                                                    lpPower=lpPower,
                                                    )
        monomials = _sanitizeMonomialsForBooleans(monomials, boolean_variables)
        replaced2 = [k / 2 * Moment(*alpha) for (k, alpha) in monomials]
        return Add(*replaced1) - Add(*replaced2)
    else:
        raise RuntimeError("Higher than 2nd order transitions are not implemented yet")

# -------------------------------------------------
def get_dfMdt(transition_classes, fM, D, order=2, 
                                         orderW=2,
                                         lpac=False,
                                         substitutions=[], 
                                         boolean_variables=set(),
                                         ):
    """
    Given a function of Moments f(M) and a set of transitions classes, compute the derivative df(M)/dt.

    :param transition_classes: list of transition classes
    :param fM: a function of Moments
    :param D: number of species
    :param lpac=False: use the LPAC approximation mode
    :param substitutions=[]: pass a list of raw custom substitutions to use
    :param boolean_variables=set(): a set of variables indices to be treated as 
    boolean. New vars will be added here to be used again in the caller.
    """
    import sys
    if _getNumSpecies(fM) != D:
        raise RuntimeError(f'Arities of all occurring moments should be {D}. ({fM})')
    dfM = ito(fM)
    # print("dfM: %s" %(dfM), file=stderr) #debug
    monomials = decomposeMomentsPolynomial(dfM)
    # print("monomials: %s" %(monomials), file=stderr) #debug
    contrib = list()
    for c, tc in enumerate(transition_classes):
        transition, k_c, g_c, pi_c = tc.transition, tc.k, tc.g, tc.pi
        isBulk = transition.isBulk
        for q, (k_q, pM, pDM) in enumerate(monomials):
            # debug(">>> (%d) fM: %s\tpM: %s\tpDM: %s" %(c,fM,pM,pDM)) #debug
            reactants, products, DM_cj = None, None, None
            if not isBulk:
                reactants = getCompartments(transition.lhs)
                products = getCompartments(transition.rhs)
                DM_cj = getDeltaM(reactants, products, D)
            else:
                reactants = transition.lhs
                products = transition.rhs
                DM_cj = products - reactants
            debug(">>> (%d,%d) pDM = %s" %(c,q,pDM)) #debug
            debug(">>> (%d,%d) DM_cj = %s" %(c,q,DM_cj)) #debug
            pDMcj = subsDeltaM(pDM, DM_cj)
            debug(">>> (%d,%d) pDMcj = %s" %(c,q,pDMcj)) #debug
            cexp = pi_c.conditional_expectation(pDMcj) #Expensive!
            contentFun = g_c * cexp
            cF = None
            while cF != contentFun:
                cF = contentFun
                contentFun = cF.expand().subs(substitutions) # Test manual simplifications

            # Discover any boolean variable in the current expression and
            # add them to the global set
            curBV = _getBooleanVariablesIndices(contentFun, D)
            boolean_variables.update(curBV)
            # Handle the boolean variables for this expression
            contentFun = _handleBooleanVariables(contentFun, D, boolean_variables)
            contextFun = k_c * k_q * pM
            dfMdt = None
            # lpac autoclosure for content function
            if lpac==True:
                #Expensive!
                dfMdt = get_dfMdt_contrib_lpac(reactants, contextFun, contentFun, D, 
                        order=order,
                        boolean_variables=boolean_variables,
                        # lpPower=2,
                        )

                debug(">>> (init) dfMdt=%s" %(dfMdt)) #debug
                # Now let's take care to expand in w
                # We first extract all the moments present in the expression
                singleMoments = getSingleMomentsInExpression(dfMdt)
                expectedMoments = getExpectedMomentsInExpression(dfMdt)
                debug("singleMoments=%s" %(singleMoments)) #debug
                debug("expectedMoments=%s" %(expectedMoments)) #debug
                if len(singleMoments) > 0: # Only do this if there is any moment to differentiate
                    momentsList = list(singleMoments.union(expectedMoments))
                    debug(">>> momentsList=%s" %(momentsList)) #debug
                    # Then we need to substitute them with dummy variables, so that
                    # the differentiation is successful
                    _x = Content('_x')
                    _Ex = Content('_Ex')
                    _Esubs = [ (Expectation(M), _Ex[i]) for i,M in enumerate(momentsList) ]
                    dfMdt = dfMdt.subs(_Esubs)
                    _subs = [ (M, _x[i]) for i,M in enumerate(momentsList) ]
                    _varList = [ _x[i] for i,M in enumerate(momentsList) ]
                    dfMdt = dfMdt.subs(_subs)
                    # Now we perform the Taylor expansion
                    debug(">>> (pre-taylor) dfMdt=%s" %(dfMdt)) #debug
                    dfMdt = _multivariateTaylor( #Expensive!
                                dfMdt,
                                _varList,
                                [ Expectation(z) for z in _varList ],
                                orderW,
                                )
                    # dfMdt = dfMdt.factor()
                    # And finally we substitute the Moments back into the expression
                    debug(">>> (post-taylor) dfMdt=%s" %(dfMdt)) #debug
                    _reverseSubs = [ (_x[i], M) for i,M in enumerate(momentsList) ]
                    dfMdt = dfMdt.subs(_reverseSubs)
                    _reverseEsubs = [ (_Ex[i], Expectation(M)) for i,M in enumerate(momentsList) ]
                    dfMdt = dfMdt.subs(_reverseEsubs)

                debug(">>> Expansion(dfMdt)=%s" %(dfMdt)) #debug
            else:
                l_n_Xc = contextFun * contentFun
                dfMdt = get_dfMdt_contrib(reactants, l_n_Xc, D, 
                        boolean_variables=boolean_variables)
            contrib.append(dfMdt)
    return Add(*contrib)


# -------------------------------------------------
def _expectation(expr):
    """
    Replace each moments expression (product of moments) in monomials of expr with
    the expectation of the moments expression.

    :param expr: a polynomial in M^{\gamma^k}.
    :return: expr with every fM replaced by Expectation(fM)
    """
    monomials = decomposeMomentsPolynomial(expr, strict=False)
    contrib = list()
    for (k, M, DM) in monomials:
        if M != 1:
            M = Expectation(M)
        if DM != 1:
            raise RuntimeError("Did not expect any deltaM in expression." + str(expr))
        contrib.append(k * M)
    return Add(*contrib)


def _verifyContentNumSpecies(content, D):
    """
    Verify that the number of species occurring in all ContentChanges in content equals D.
    Raise an error otherwise.

    :param Expr content: the content of the compartment, comprising Contents, ContentChanges, sums of those, and multiplication by integers
    :param int D: the number of species
    """
    if content.func in [Add, Mul]:
        for arg in content.args:
            _verifyContentNumSpecies(arg, D)
    elif content.func in [ContentChange, Moment]:
        if len(content.args) != D:
            raise RuntimeError("Number of species occurring in in ContentChange expression"
                               + str(content) + " is " + str(len(content.args))
                               + ". Expected " + str(D))
    elif issubclass(content.func, Integer):
        pass
    elif content.func == Content:
        pass
    else:
        raise TypeError("Unexpected expression " + str(content))


def _getAndVerifyNumSpecies(transition_classes, moments, D=None):
    """
    Verify that the number of species occurring in all transitions and moment expressions equals D.
    Raise an error otherwise.
    If D is not specified, just checks that number of species occurring in all transitions and
    moment expressions is the same

    :param transition_classes: a list of TransitionClass
    :param moments: a list of Moment expressions
    :param D: optionally, the expected number of species
    :return: the number of species
    """
    for fM in moments:
        DfM = _getNumSpecies(fM)
        if D is None:
            D = DfM
        if D != DfM:
            raise RuntimeError("Number of species occurring in in Moment expressions is not unique."
                               + str(fM) + " contains moments of orders " + str(DfM)
                               + ". Expected order " + str(D))
    for tc in transition_classes:
        if not tc.transition.isBulk:
            for c in getCompartments(tc.transition.lhs).keys():
               _verifyContentNumSpecies(c.args[0], D)
            for c in getCompartments(tc.transition.rhs).keys():
               _verifyContentNumSpecies(c.args[0], D)
        else:
            _verifyContentNumSpecies(tc.transition.lhs, D)
            _verifyContentNumSpecies(tc.transition.rhs, D)
    return D

###################################################
#
# "Outer loop": get missing moments and iterate
#
###################################################

def getKnownMoments(RHSset):
    known_moments = set()
    for RHS in RHSset:
        known_moments = known_moments.union(getMomentsInExpression(RHS))
    return known_moments

def apply_substitutions(equations, substitutions):
    """
    Perform raw term substitutions.

    :param equations: list of pairs (fM, dfMdt). (as returned by compute_moment_equations)
    :param substitutions: list of tuples of (expr, substitution(expr))
    :return: list of pairs (fM, dfMdt'), where dfMdt' is obtained by substituting closed moments into dfMdt
    """
    substitutions = {m: c for m, c in substitutions}
    # print("Substitutions: %s" %(substitutions)) #debug
    subs_equations = [(fM, dfMdt.subs(substitutions)) for fM, dfMdt in equations]
    return subs_equations


def compute_moment_equations(transition_classes, moments, 
                                substitutions=[], D=None, 
                                order=2,
                                orderW=2,
                                lpac=False,
                                boolean_variables=set(),
                                simplify_equations=False,
                                ):
    """
    Given a reaction network, moment expressions, and number of species, computes
    a list of pairs `(fM, dfMdt)`, where each pair consists of the desired moment expression
    and the derived expression for its time derivative.

    :param transition_classes: list of all transition classes of the reaction network
    :param moments: a list of functions of Moments
    :param substitutions: list of tuples of (expr, substitution(expr)) to be substituted before taking the expectation
    :param D: optionally, the number of species
    :return: list of pairs (fM, dfMdt)
    """
    debug("> Compute Moment Equations: moments=%s" %(moments))
    info("> Compute Moment Equations: computing equations for %d moments" %(len(moments)), 
        flush=True)
    t0 = time()
    D = _getAndVerifyNumSpecies(transition_classes, moments, D)
    equations = list()
    required = set()
    if type(lpac)!=list:
        lpac = [ lpac for x in moments ]
    elif len(lpac)<len(moments):
        lpac = [ lpac[i] if i<len(lpac) else True for i in range(len(moments)) ]
    for (fM,curlpac) in zip(moments, lpac):
        dfMdt = get_dfMdt(transition_classes, fM, D, 
                            order=order,
                            orderW=orderW,
                            lpac=curlpac, 
                            substitutions=substitutions,
                            boolean_variables=boolean_variables,
                            )
        dfMdt = dfMdt.expand().subs(substitutions)
        EdfMdt = _expectation(dfMdt).expand()
        if simplify_equations:
            EdfMdt = EdfMdt.simplify()
        equations.append(( fM, EdfMdt ))
    dt = time() - t0
    info(" [%s]" %(str(timedelta(seconds=dt))))
    return equations


def get_missing_moments(equations, known_moments=None):
    """
    Given a system of moment equations, compute the moment expressions that occur
    on the right-hand-side of equations but are not governed by the system.

    :param equations: a list of pairs `(fM, dfMdt)`, where each pair consists of
            the desired moment expression and the derived expression for its time derivative.
    :return: set of missing moment expressions
    """

    def _get_moment_epressions(expr):
        if expr.func is Expectation:
            debug(">>> _get_moment_epressions: expr = %s" %(expr))
            return {expr.args[0]}
        elif expr.func in [Add, Mul, Pow]:
            return set(itertools.chain(*(_get_moment_epressions(arg) for arg in expr.args)))
        else:
            # debug(">>> _get_moment_epressions: expr = %s" %(expr))
            return {}

    rhs = set(itertools.chain(*(_get_moment_epressions(dfMdt) for _, dfMdt in equations)))
    lhs = known_moments
    if lhs == None:
        lhs = set(fM for fM, _ in equations)
    else:
        lhs = set(lhs)
    return rhs - lhs
