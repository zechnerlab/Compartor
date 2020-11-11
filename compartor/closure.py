from sympy import Mul, Pow, Integer
from compartor.compartments import Moment, Expectation
import itertools


# -------------------------------------------------
def __getMomentPowers(expr):
    """
    Decompose a product of powers of moments to list [(Moment, power)]

    :param expr: product of powers of Moments
    :return: list of tuples (Moment, power)
    """
    if expr.func == Mul:
        moments = [__getMomentPowers(arg) for arg in expr.args]
        return list(itertools.chain(*moments))
    elif expr.func == Pow and expr.args[0].func == Moment and issubclass(expr.args[1].func, Integer):
        return [(expr.args[0], expr.args[1])]
    elif expr.func == Moment:
        return [(expr, 1)]
    else:
        raise TypeError("Unexpected expression " + str(expr))


# -------------------------------------------------
def gamma_closure(expr):
    """
    Try to compute gamma closure for expectation of expr.

    :param expr: product of powers of moments
    :return: gamma closure of <expr>
    """

    def __closure1(moment_powers):
        moment = moment_powers[0][0]
        if 3 in moment.args:
            i = moment.args.index(3)
            j = i
        else:
            i = moment.args.index(2)
            j = moment.args.index(1)

        def m(a, b):
            gamma = [0] * len(moment.args)
            gamma[i] += a
            gamma[j] += b
            return Expectation(Moment(*gamma))

        expr = 2 * m(2, 0) * m(1, 1) / m(1, 0) - m(2, 0) * m(0, 1) / m(0, 0)
        return expr

    def __closure2(moment_powers):
        m_xi = moment_powers[0][0]
        if len(moment_powers) == 1:
            m_gamma = m_xi
        else:
            m_gamma = moment_powers[1][0]
        expr = 2 * (Expectation(m_gamma ** 2) * Expectation(m_gamma * m_xi)) / Expectation(m_gamma) - Expectation(
            m_gamma ** 2) * Expectation(m_xi)
        return expr

    # get tuples(M, pow) in expr
    moment_powers = __getMomentPowers(expr)
    # sort by ascending pow
    moment_powers = sorted(moment_powers, key=lambda x: x[1])

    if len(moment_powers) == 1 and moment_powers[0][1] == 1 and moment_powers[0][0].order() == 3:
        return __closure1(moment_powers)
    elif len(moment_powers) == 1 and moment_powers[0][1] == 3:
        return __closure2(moment_powers)
    elif len(moment_powers) == 2 and moment_powers[0][1] == 1 and moment_powers[1][1] == 2:
        return __closure2(moment_powers)
    else:
        raise ValueError("Could not derive Gamma closure for " + str(expr) + " (Not implemented yet).")


# -------------------------------------------------
def gamma_closures(expressions, could_not_close=None):
    """
    Try to compute gamma closure for each of expressions.

    :param expressions: list of moment expressions, where each expression is a product of
            powers of moments
    :param could_not_close: a list that will be filled with expressions that could not be closed.
            If this optional argument is not specified, then a ValueError will be raised if an
            expression cannot be closed.
    :return: list of tuples of (expr, gamma_closure(expr))
    """
    closures = []
    for expr in expressions:
        try:
            closure = gamma_closure(expr)
            closures.append((expr, closure))
        except ValueError:
            if could_not_close is not None:
                could_not_close.append(expr)
            else:
                raise
    return closures


# -------------------------------------------------
def meanfield_closure(expr):
    """
    Compute meanfield closure for expectation of expr.

    :param expr: product of powers of moments
    :return: meanfield closure of <expr>
    """

    def __product(list):
        r = 1
        for x in list:
            r *= x
        return r

    # get tuples(M, pow) in expr
    moment_powers = __getMomentPowers(expr)
    # sort by ascending pow
    moment_powers = sorted(moment_powers, key=lambda x: x[1])

    return  __product([Expectation(moment_power[0])**moment_power[1] for moment_power in moment_powers])


# -------------------------------------------------
def meanfield_closures(expressions):
    """
    Compute meanfield closure for each of expressions.

    :param expressions: list of moment expressions, where each expression is a product of powers of moments
    :return: list of tuples of (expr, meanfield_closure(expr))
    """
    return [(expr, meanfield_closure(expr)) for expr in expressions]


# -------------------------------------------------
def hybrid_closures(expressions):
    """
    Attempt gamma closure for each of expressions.
    For the expressions that could not be closed by gamma closure, do mean-field closure
    """
    could_not_close = []
    gamma = gamma_closures(expressions, could_not_close)
    meanfield = meanfield_closures(could_not_close)
    return gamma+meanfield


# -------------------------------------------------
def substitute_closures(equations, closures):
    """
    Take expectation of rhs in evolutions, and substitute closures.

    :param equations: list of pairs (fM, dfMdt). (as returned by compute_moment_equations)
    :param closures: list of tuples of (expr, closure(expr))
    :return: list of pairs (fM, dfMdt'), where dfMdt' is obtained by substituting closed moments into dfMdt
    """
    substitutions = {Expectation(m): c for m, c in closures}
    closed_equations = [(fM, dfMdt.subs(substitutions)) for fM, dfMdt in equations]
    return closed_equations
