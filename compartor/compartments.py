from sympy import Function, IndexedBase, Indexed, Basic, Symbol, EmptySet, Add, Mul, Pow, Integer, Eq, KroneckerDelta, \
    factorial, ff
from sympy.core.decorators import call_highest_priority
import itertools
import collections


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

    def __new__(cls, lhs, rhs, name=None):
        t = Basic.__new__(cls)
        t.lhs = lhs
        t.rhs = rhs
        t.name = name
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
            return Sum(
                pDMcj * 1 / (end - start + 1),
                (y, start, end)
            ).doit().factor().expand()

        return OutcomeDistribution(symbol, expectation)

    #@classmethod
    #def Binomial(cls, symbol, y, n, p):
    #    """
    #    Returns an OutcomeDistribution that is a Binomial distribution
    #
    #    :param symbol: symbol to use when displaying Pi_c in equations
    #    :param y: random variable, entry in a content variable, e.g., y[0]
    #    :param n: number parameter of the Binomial distribution
    #    :param p: success probability of the Binomial distribution.
    #    :return Pi_c:
    #    """
    #    from sympy.stats import Binomial, E
    #    def expectation(pDMcj):
    #        binomial = Binomial('binomial', n, p)
    #        return E(pDMcj.subs(y, binomial))
    #
    #    return OutcomeDistribution(symbol, expectation)


###################################################
#
# Specifying transitions classes
#
###################################################

# -------------------------------------------------
class TransitionClass(Basic):
    """
    Transition class comprising a Transition, a content-independent rate constant k, a reactant tuning function g, and the outcome distribuiton pi
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
    if expr.func in [Add, Mul, Compartment, ContentChange]:
        return set(itertools.chain(*(_getContentVars(arg) for arg in expr.args)))
    elif expr.func == Content:
        return {expr}
    elif expr.func == Indexed:
        return {expr.base}
    elif expr is EmptySet:
        return set()
    elif issubclass(expr.func, Integer):
        return set()
    else:
        raise TypeError("Unexpected expression " + str(expr))


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
def __decomposeContentPolynomial(expr, x, D):
    """
    Given a polynomial in Xc = {x}, decompose its monomials as (constant * prod x[i]^alpha[i])

    :param expr: a polynomial in Xc = {x}.
    :param x: content variable x
    :param D: number of species
    :return: list of monomials, each decomposed into (constant, alpha)
    """
    expr = expr.factor().expand()
    monomials = list(expr.args) if expr.func == Add else [expr]
    result = list()
    for monomial in monomials:
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
        result.append((k, alpha))
    return result


# -------------------------------------------------
def __decomposeContentPolynomial2(expr, x, y, D):
    """
    Given a polynomial in Xc = {x, y}, decompose its monomials as (constant * prod x[i]^alpha[i] * prod y[i]^beta[i])

    :param expr: a polynomial in Xc = {x,y}.
    :param x: content variable x
    :param y: content variable y
    :param D: number of species
    :return: list of monomials, each decomposed into (constant, alpha, beta)
    """
    expr = expr.expand()
    monomials = list(expr.args) if expr.func == Add else [expr]
    result = list()
    for monomial in monomials:
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
        raise RuntimeError("Cannot determine number of species from expression." + str(expr))
    elif len(Ds) == 1:
        return next(iter(Ds))
    else:
        raise RuntimeError("Number of species in Moments occurring in expression is not unique."
                           + str(expr) + " contains moments of orders " + str(Ds))


# -------------------------------------------------
def get_dfMdt_contrib(reactants, l_n_Xc, D):
    """
    Compute the contribution to df(M)/dt of a particular transition and a particular monomial.

    :param reactants:
    :param l_n_Xc:
    :param D:
    :return:
    """
    if len(reactants) == 0:
        return l_n_Xc
    elif len(reactants) == 1:
        (compartment, count) = next(iter(reactants.items()))
        __checkSimpleCompartment(compartment)
        if count != 1:
            raise RuntimeError("not implemented yet")
        # case Xc=={x}
        # compartment==[x]
        x = compartment.args[0]
        monomials = __decomposeContentPolynomial(l_n_Xc, x, D)
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
        # case Xc=={x, x'}
        # compartment1==[x]
        # compartment2==[x']
        x = compartment1.args[0]
        x1 = compartment2.args[0]
        monomials = __decomposeContentPolynomial2(l_n_Xc, x, x1, D)
        replaced1 = [k / 2 * Moment(*alpha) * Moment(*beta) for (k, alpha, beta) in monomials]
        monomials = __decomposeContentPolynomial(l_n_Xc.subs(x1, x), x, D)
        replaced2 = [k / 2 * Moment(*alpha) for (k, alpha) in monomials]
        return Add(*replaced1) - Add(*replaced2)
    else:
        raise RuntimeError("Higher than 2nd order transitions are not implemented yet")


# -------------------------------------------------
def get_dfMdt(transition_classes, fM, D):
    """
    Given a function of Moments f(M) and a set of transitions classes, compute the derivative df(M)/dt.

    :param transition_classes: list of transition classes
    :param fM: a function of Moments
    :param D: number of species
    """
    if _getNumSpecies(fM) != D:
        raise RuntimeError(f'Arities of all occurring moments should be {D}. ({fM})')
    dfM = ito(fM)
    monomials = decomposeMomentsPolynomial(dfM)
    contrib = list()
    for c, tc in enumerate(transition_classes):
        transition, k_c, g_c, pi_c = tc.transition, tc.k, tc.g, tc.pi
        for q, (k_q, pM, pDM) in enumerate(monomials):
            reactants = getCompartments(transition.lhs)
            products = getCompartments(transition.rhs)
            DM_cj = getDeltaM(reactants, products, D)
            pDMcj = subsDeltaM(pDM, DM_cj)
            cexp = pi_c.conditional_expectation(pDMcj)
            l_n_Xc = k_c * k_q * pM * g_c * cexp
            dfMdt = get_dfMdt_contrib(reactants, l_n_Xc, D)
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
    elif content.func == ContentChange:
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
        for c in getCompartments(tc.transition.lhs).keys():
           _verifyContentNumSpecies(c.args[0], D)
        for c in getCompartments(tc.transition.rhs).keys():
           _verifyContentNumSpecies(c.args[0], D)
    return D


###################################################
#
# "Outer loop": get missing moments and iterate
#
###################################################

def getRequiredMoments(dfMdt):
    monomials = decomposeMomentsPolynomial(dfMdt, strict=False)
    required = set()
    for (k, M, DM) in monomials:
        if M != 1:
            required.add(M)
    return required


def compute_moment_equations(transition_classes, moments, D=None):
    """
    Given a reaction network, moment expressions, and number of species, computes
    a list of pairs `(fM, dfMdt)`, where each pair consists of the desired moment expression
    and the derived expression for its time derivative.

    :param transition_classes: list of all transition classes of the reaction network
    :param moments: a list of functions of Moments
    :param D: optionally, the number of species
    :return: list of pairs (fM, dfMdt)
    """
    D = _getAndVerifyNumSpecies(transition_classes, moments, D)
    equations = list()
    required = set()
    for fM in moments:
        dfMdt = get_dfMdt(transition_classes, fM, D)
        equations.append((fM, _expectation(dfMdt)))
    return equations


def get_missing_moments(equations):
    """
    Given a system of moment equations, compute the moment expressions that occur
    on the right-hand-side of equations but are not governed by the system.

    :param equations: a list of pairs `(fM, dfMdt)`, where each pair consists of
            the desired moment expression and the derived expression for its time derivative.
    :return: set of missing moment expressions
    """

    def _get_moment_epressions(expr):
        if expr.func is Expectation:
            return {expr.args[0]}
        elif expr.func in [Add, Mul, Pow]:
            return set(itertools.chain(*(_get_moment_epressions(arg) for arg in expr.args)))
        else:
            return {}

    rhs = set(itertools.chain(*(_get_moment_epressions(dfMdt) for _, dfMdt in equations)))
    lhs = set(fM for fM, _ in equations)
    return rhs - lhs
