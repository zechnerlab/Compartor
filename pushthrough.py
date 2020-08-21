from sympy import EmptySet, Add, Mul, Pow, Integer, Indexed, Symbol
from compartmentsBase import *
from compartmentsB import pi_c_identity, pi_c_poisson

D = 1 # number of species

y = ContentVar('y')
x = ContentVar('x')

# Intake
transition_I = Transition(EmptySet(), Compartment(y))
k_I = Constant('k_I')
g_I = 1
pi_I = pi_c_poisson(
    Symbol("\pi_I", positive=True),
    y[0],
    Symbol("\lambda", positive=True))

# Exit
transition_E = Transition(Compartment(x), EmptySet())
k_E = Constant('k_E')
g_E = 1
pi_E = pi_c_identity()

# birth
transition_b = Transition(Compartment(x), Compartment(x + ContentChange(1)))
k_b = Constant('k_b')
g_b = 1
pi_b = pi_c_identity()

# death
transition_d = Transition(Compartment(x), Compartment(x + ContentChange(-1)))
k_d = Constant('k_d')
g_d = x[0] # TODO x should be enough here, in case D=1.
pi_d = pi_c_identity()

transitions = [
    (transition_I, k_I, g_I, pi_I),
    (transition_E, k_E, g_E, pi_E),
    (transition_b, k_b, g_b, pi_b),
    (transition_d, k_d, g_d, pi_d)
]


# ================================================================

from ito import ito, decomposeMomentsPolynomial

dM1M1 = ito(Moment(1)**2)
monomials = decomposeMomentsPolynomial(dM1M1)

print(dM1M1)
print(monomials)
print()


from compartmentsB import getCompartments, deltaM, subsDeltaM, get_dfMdt_contrib

c = 2
q = 0
print(f'c = {c}, q = {q}')

(transition, k_c, g_c, pi_c) = transitions[c]
(k_q, pM, pDM) = (monomials[q].k, monomials[q].pM, monomials[q].pDM)
print(f'pDM = {pDM}')

reactants = getCompartments(transition.lhs)
products = getCompartments(transition.rhs)

DM_cj = deltaM(reactants, products, D)
print(f'DM_cj = {DM_cj}')

pDMcj = subsDeltaM(pDM, DM_cj)
print(f'pDMcj = {pDMcj}')

# TODO
def conditional_expectation(pDMcj, pi_c):
    return pDMcj # should be pi_c(pDMcj, ???)

cexp = conditional_expectation(pDMcj, pi_c)
l_n_Xc = k_c * k_q * pM * g_c * cexp
print(f'l_n_Xc = {l_n_Xc}')

dfMt = get_dfMdt_contrib(reactants, l_n_Xc, D)
print(f'dfMt = {dfMt}')

# -------------------------------------------------

print()
print("------------------------------------------------")
print()

# -------------------------------------------------


def get_dfMdt(transitions, fM, D):
    dfM = ito(fM)
    monomials = decomposeMomentsPolynomial(dfM)
    contrib = list()
    for c, (transition, k_c, g_c, pi_c) in enumerate(transitions):
        for q, (k_q, pM, pDM) in enumerate([(m.k, m.pM, m.pDM) for m in monomials]):
            reactants = getCompartments(transition.lhs)
            products = getCompartments(transition.rhs)
            DM_cj = deltaM(reactants, products, D)
            pDMcj = subsDeltaM(pDM, DM_cj)
            cexp = pi_c.conditional_expectation(pDMcj)
            l_n_Xc = k_c * k_q * pM * g_c * cexp
            dfMdt = get_dfMdt_contrib(reactants, l_n_Xc, D)
            contrib.append(dfMdt)
            #print(f'c={c}, q={q}')
            #display(Eq(symbols('\\text{dfMdt}'), dfMdt))
    return Add(*contrib)

dNdt = get_dfMdt(transitions, Moment(0), 1)
print(f'dNdt = {dNdt}')
