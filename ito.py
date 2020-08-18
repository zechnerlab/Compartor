from compartmentsBase import Moment, DeltaM
from sympy import Function, Add, Mul, Pow, Integer
import itertools


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
    Apply Ito's rule to a function of moments.
    :param expr: expression comprising Moments, addition, multiplication, and powers with moments only in the base
    :return: expansion by Ito's rule
    """
    moments = __getMoments(expr)
    substitutions = [(m, m + DeltaM(*m.args)) for m in moments]
    expr = expr.subs(substitutions) - expr
    return expr.expand()


# -------------------------------------------------
class MomentsMonomial(object):
    """
    This class represents a decomposed monomial in M^{\gamma^k} and \DeltaM^{\gamma^l}.
    It comprises a constant k, a product of moments pM, and a product of moment updates pDM.
    """
    def __init__(self, k, pM, pDM):
        self.k = k
        self.pM = pM
        self.pDM = pDM

    def __repr__(self):
        return f'MomentsMonomial({self.k}, {self.pM}, {self.pDM})'


# -------------------------------------------------
def decomposeMomentsPolynomial(expr):
    """
    :param expr: a polynomial in M^{\gamma^k} and \DeltaM^{\gamma^l}.
    :return: list of monomials, each decomposed into (constant, product of Moments, product of DeltaMoments)
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
            elif factor.func == Pow and issubclass(factor.args[ 1 ].func, Integer):
                if factor.args[ 0 ].func == Moment:
                    qM *= factor
                elif factor.args[ 0 ].func == DeltaM:
                    qDM *=factor
                else:
                    raise TypeError("Unexpected expression " + str(factor))
            else:
                raise TypeError("Unexpected expression " + str(factor))
        result.append(MomentsMonomial(qK, qM, qDM))
    return result




# --- demo ---
if __name__ == "__main__":
    m0 = Moment(0)
    expr = m0*m0
    print(expr)
    print(ito(expr))
    print(decomposeMomentsPolynomial(ito(expr)))
