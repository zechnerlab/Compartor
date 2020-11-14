from sympy import Add, Mul, Pow, Number, Symbol, simplify
from compartor.compartments import Moment, Expectation, compute_moment_equations, get_missing_moments, _getNumSpecies, _getAndVerifyNumSpecies
from compartor.closure import gamma_closures, meanfield_closures, hybrid_closures, substitute_closures, __getMomentPowers

import itertools
import collections

def automated_moment_equations(D, transition_classes, moments=None):
    """
    Outputs a closed system of moment equations for the provided transition classes.
    The moments to be characterized are identified automatically to comprise at least expected number and total mass dynamics.
    Optionally, a desired set of moments to be characterized can be passed as third argument.
    When necessary, third-order Gamma closures and possibly mean-field closures are applied.

    :param D: the number of chemical species
    :param transition_classes: a list of TransitionClass
    :param moments: optionally, a list of Moment expressions
    :return: list of included moments, moment equations
    """
    if moments is None:
        indexes = [0 for d in range(D)]
        moments = [Moment(*tuple(indexes))]
        for d in range(D):
            indexes[d]+=1
            moments.append(Moment(*tuple(indexes)))
            indexes[d]-=1
    D = _getAndVerifyNumSpecies(transition_classes, moments, D)
    iterate = True
    while iterate:
        equations = compute_moment_equations(transition_classes, moments)
        missing = get_missing_moments(equations)
        if len(missing)==0:
            iterate = False
        else:
            not_closed=[]
            closures = gamma_closures(missing, not_closed)
            if len(not_closed)==0:
                equations = substitute_closures(equations, closures)
                iterate = False
            elif len(get_worth_adding(not_closed)) > 0:
                moments_to_add = get_worth_adding(not_closed)
                for mom in moments_to_add:
                    moments.append(mom)
            else:
                closures = hybrid_closures(missing)
                equations = substitute_closures(equations, closures)
                iterate = False

    return moments, equations


def _worth_adding(expr):
    moment_powers = __getMomentPowers(expr)
    # sort by ascending pow
    moment_powers = sorted(moment_powers, key=lambda x: x[1])
    if len(moment_powers) < 3 and sum([moment_powers[i][1] for i in range(len(moment_powers))]) < 3 and sum([moment_powers[i][0].order() for i in range(len(moment_powers))]) < 3:
        return True
    else:
        return False

def get_worth_adding(expressions):
    worth = []
    for expr in expressions:
        if _worth_adding(expr):
            worth.append(expr)
    return worth
