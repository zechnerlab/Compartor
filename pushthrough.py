from sympy import EmptySet
from compartmentsBase import *

transition_E = Transition(Compartment(ContentVar('x')), EmptySet())
transition_b = Transition(Compartment(ContentVar('x')), Compartment(ContentVar('x') + ContentChange(1)))
transition_d = Transition(Compartment(ContentVar('x')), Compartment(ContentVar('x') + ContentChange(-1)))

k_E = Constant('k_E')
g_E = n(ContentVar('x'))
pi_E = 1

k_b = Constant('k_b')
g_b = n(ContentVar('x'))
pi_b = 1

k_d = Constant('k_d')
g_d = n(ContentVar('x')) * ContentVar('x')[0]    # TODO ContentVar('x') should be enough here, in case D=1.
pi_d = 1

