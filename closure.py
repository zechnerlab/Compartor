from sympy import Add, Mul, Pow, Integer
from compartments import Moment, Expectation, _expectation
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

    try:
        if len(moment_powers) == 1 and moment_powers[0][1] == 1 and moment_powers[0][0].order() == 3:
            return __closure1(moment_powers)
        elif len(moment_powers) == 1 and moment_powers[0][1] == 3:
            return __closure2(moment_powers)
        elif len(moment_powers) == 2 and moment_powers[0][1] == 1 and moment_powers[1][1] == 2:
            return __closure2(moment_powers)
    except ValueError:
        pass
    raise RuntimeError("Not implemented yet: closure for " + str(expr))


# -------------------------------------------------
def gamma_closures(expressions):
    """
    Try to compute gamma closure for each of expressions.

    :param expressions: list of moment expressions, where each expression is a product of powers of moments
    :return: list of tuples of (expr, gamma_closure(expr))
    """
    return [(expr, gamma_closure(expr)) for expr in expressions]


# -------------------------------------------------
def substitute_closures(evolutions, closures):
    """
    Take expectation of rhs in evolutions, and substitute closures.

    :param evolutions: list of pairs (fM, dfMdt). (as returned by compute_moment_evolutions)
    :param closures: list of tuples of (expr, closure(expr))
    :return: list of pairs (fM, dfMdt'), where dfMdt' is obtained by taking the expectation of dfMdt and substituting
    closed moments
    """
    substitutions = {Expectation(m): c for m, c in closures}
    closed_evolutions = [(fM, dfMdt.subs(substitutions)) for fM, dfMdt in evolutions]
    return closed_evolutions
