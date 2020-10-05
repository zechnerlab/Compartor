from sympy import KroneckerDelta, factorial, ff
from sympy import Eq, Symbol, Function, Derivative, Add
from compartments import Moment, Compartment, Expectation, __checkSimpleCompartment, getCompartments, decomposeMomentsPolynomial
from IPython.core.display import display


###################################################
#
# Displaying transitions in notebooks
#
###################################################

# -------------------------------------------------
def __n(content):
    """
    Expression for number of compartments with given content
    """
    if content.func == Compartment:
        return __n(content.args[0])
    return Function('n', integer=True)(content)


# -------------------------------------------------
def __kronecker(content1, content2):
    if content1.func == Compartment:
        return __kronecker(content1.args[0], content2)
    if content2.func == Compartment:
        return __kronecker(content1, content2.args[0])
    return KroneckerDelta(content1, content2)


# -------------------------------------------------
def __getWnXc(reactants):
    """
    Get w(n;Xc)

    :param dict reactants: reactant compartments Xc as a dictionary that maps Compartment to number of occurrences
    :return: w(n;Xc)
    """
    if len(reactants) == 0:
        return 1
    elif len(reactants) == 1:
        (compartment, count) = next(iter(reactants.items()))
        __checkSimpleCompartment(compartment)
        return 1 / factorial(count) * ff(__n(compartment), count)
    elif len(reactants) == 2:
        i = iter(reactants.items())
        (compartment1, count1) = next(i)
        (compartment2, count2) = next(i)
        __checkSimpleCompartment(compartment1)
        __checkSimpleCompartment(compartment2)
        if count1 != 1 or count2 != 1:
            raise RuntimeError("Higher than 2nd order transitions are not implemented yet")
        return __n(compartment1) * (__n(compartment2) - __kronecker(compartment1, compartment2)) \
               / ( 1 + __kronecker(compartment1, compartment2))
    else:
        raise RuntimeError("Higher than 2nd order transitions are not implemented yet")


# -------------------------------------------------
def display_propensity(transition, name=None):
    (tr, k, g, pi) = transition
    if name is None:
        name = tr.name
    if name is None:
        name = ''
    h_c = Symbol("h_{" + name + "}")
    reactants = getCompartments(tr.lhs)
    w = __getWnXc(reactants)
    expr = k * g * pi.expr * w
    display(Eq(h_c, expr, evaluate=False))


# -------------------------------------------------
def display_propensity_details(transition, name=None):
    (tr, k, g, pi) = transition
    if name is None:
        name = tr.name
    if name is None:
        name = ''
    display(Eq(Symbol('k_{' + name + '}'), k, evaluate=False))
    display(Eq(Symbol('g_{' + name + '}'), g, evaluate=False))
    display(Eq(Symbol('\pi_{' + name + '}'), pi.expr, evaluate=False))


# -------------------------------------------------
def display_transitions(transitions, details=False):
    for t in transitions:
        display(t[0])
        display_propensity(t)
        if details is True:
            display_propensity_details(t)


###################################################
#
# Displaying expected moment evolution in notebooks
#
###################################################

# -------------------------------------------------
def display_expected_moment_evolution(expr_moment, expr_dfMdt, D=None):
    """
    :param expr_moment: lhs of evolution equation
    :param expr_dfMdt: rhs of evolution equation
    :param D: if present, Moment([0[*D) will be replaced by N
    """
    lhs = Derivative(Expectation(expr_moment), Symbol('t'))
    monomials = decomposeMomentsPolynomial(expr_dfMdt, strict=False)
    contrib = list()
    for (k, M, DM) in monomials:
        if M != 1:
            M = Expectation(M)
        if DM != 1:
            raise RuntimeError("Did not expect any deltaM in expression." + str(expr_dfMdt))
        contrib.append(k * M * DM)
    rhs = Add(*contrib)
    evolution = Eq(lhs, rhs, evaluate=False)
    if not D is None:
        evolution = evolution.subs(Moment(*([0] * D)),Symbol('N'))
    display(evolution)


def display_expected_moment_evolutions(evolutions, D=None):
    for (fM, dfMdt) in evolutions:
        display_expected_moment_evolution(fM, dfMdt, D)
