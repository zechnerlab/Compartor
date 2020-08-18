from sympy import EmptySet
from compartmentsBase import *

D = 1

transition_E = Transition(Compartment(ContentVar('x')), EmptySet())
transition_b = Transition(Compartment(ContentVar('x')), Compartment(ContentVar('x') + ContentChange(1)))
transition_d = Transition(Compartment(ContentVar('x')), Compartment(ContentVar('x') + ContentChange(-1)))

k_E = Constant('k_E')
g_E = 1
pi_E = 1

k_b = Constant('k_b')
g_b = 1
pi_b = 1

k_d = Constant('k_d')
g_d = ContentVar('x')[0]    # TODO ContentVar('x') should be enough here, in case D=1.
pi_d = 1

transitions = [
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


from compartmentsB import getCompartments, deltaM, substituteGamma

c = 1
q = 0

(transition, k_c, g_c, pi_c) = transitions[c]
(k_q, pM, pDM) = (monomials[q].k, monomials[q].pM, monomials[q].pDM)

reactants = getCompartments(transition.lhs)
products = getCompartments(transition.rhs)

