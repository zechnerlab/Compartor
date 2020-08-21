from sympy import Function, IndexedBase, Indexed, Basic, Symbol, EmptySet, Add, Mul, Pow, Integer
import itertools
import collections


###################################################
#
# Specifying transitions
#
###################################################

# -------------------------------------------------
def ContentVar(name):
    """
    Create a content variable.
    If X is a ContentVar, then X[i] refers to content for species i.
    """
    return IndexedBase(name, integer=True, shape=1)


# -------------------------------------------------
class ContentChange(Function):
    """
    An integer vector that can be added to ContentVar to express chemical modifications.
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

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f'{self.lhs} ---> {self.rhs})'

    def _latex(self, printer=None):
        # Always use printer.doprint() otherwise nested expressions won't
        # work. See the example of ModOpWrong.
        l = printer.doprint(self.lhs)
        r = printer.doprint(self.rhs)
        return l + '\longrightarrow{}' + r


###################################################
#
# Specifying propensities
#
###################################################

# -------------------------------------------------
def Constant(name):
    return Symbol(name, real=True, constant=True)


# -------------------------------------------------
class Pi_c(object):
    """
    Represents the probability distribution \pi_c() as
        - an expression `expr` to be used when displaying in equations (typically just a symbol \pi_c)
        - a function `conditional_expectation` that, given an expression computes its expectation (over Y_c)
    """

    def __init__(self, expr, conditional_expectation):
        self.expr = expr
        self.conditional_expectation = conditional_expectation

    def __repr__(self):
        return f'Pi_c({self.expr}, {self.conditional_expectation})'


# -------------------------------------------------
def pi_c_identity():
    """
    Returns a Pi_c with identity conditional_expectation.
    This can be used for Y_c = {} or, more precisely,
    if the all content variables occurring in product compartments already occur in reactant compartments.
    """
    return Pi_c(1, lambda x: x)


# -------------------------------------------------
def pi_c_poisson(symbol, y, rate):
    """
    Returns a Pi_c that is a Poisson distribution of y

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

    return Pi_c(symbol, expectation)


# -------------------------------------------------
def pi_c_uniform(symbol, y, start, end):
    """
    Returns a Pi_c that is a uniform distribution of y with values from start (inclusive) to end (inclusive)

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

    return Pi_c(symbol, expectation)


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
# NEXT
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
        elif expr.func == EmptySet:
            continue
        else:
            raise TypeError("Unexpected expression " + str(expr))
        compartments[compartment] += count

    return compartments


# -------------------------------------------------
def decomposeMomentsPolynomial(expr):
    """
    Split a polynomial in M^{\gamma^k} and \DeltaM^{\gamma^l} into a list of monomials.

    :param expr: a polynomial in M^{\gamma^k} and \DeltaM^{\gamma^l}.
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
                else:
                    raise TypeError("Unexpected expression " + str(factor))
            else:
                raise TypeError("Unexpected expression " + str(factor))
        result.append((qK, qM, qDM))
    return result


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
    Checks that expr is a Compartment(IndexedBase) and throws TypeError if it is not
    """
    if not (expr.func == Compartment and len(expr.args) == 1 and expr.args[0].func == IndexedBase):
        raise TypeError(
            "Only compartments comprising a singe content variable are supported (not '" + str(expr) + "')")


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
        monomials = __decomposeContentPolynomial(l_n_Xc.subs(x, x1), x, D)
        replaced2 = [k / 2 * Moment(*alpha) for (k, alpha) in monomials]
        return Add(*replaced1) - Add(*replaced2)
    else:
        raise RuntimeError("Higher than 2nd order transitions are not implemented yet")


# -------------------------------------------------
def get_dfMdt(transitions, fM, D):
    """
    Given a function of Moments f(M) and a set of transitions, compute the derivative df(M)/dt.

    :param transitions: list of all transitions, where each transition is represented by a tuple
        (transition, k_c, g_c, pi_c) with a Transition transition, expressions k_c and g_c, and a Pi_c pi_c
    :param fM: a function of Moments
    :param D: number of species
    """
    dfM = ito(fM)
    monomials = decomposeMomentsPolynomial(dfM)
    contrib = list()
    for c, (transition, k_c, g_c, pi_c) in enumerate(transitions):
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
