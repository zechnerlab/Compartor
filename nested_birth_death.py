from sympy import Symbol
from compartments import Content, Transition, EmptySet, Compartment, Constant, pi_c_poisson, pi_c_identity, ContentChange

D = 1 # number of species

y = Content('y')
x = Content('x')

# Intake
transition_I = Transition(EmptySet, Compartment(y), name='I')
k_I = Constant('k_I')
g_I = 1
pi_I = pi_c_poisson(
    Symbol("\pi_I", positive=True),
    y[0],
    Symbol("\lambda", positive=True))

# Exit
transition_E = Transition(Compartment(x), EmptySet, name='E')
k_E = Constant('k_E')
g_E = 1
pi_E = pi_c_identity()

# birth
transition_b = Transition(Compartment(x), Compartment(x + ContentChange(1)), name='b')
k_b = Constant('k_b')
g_b = 1
pi_b = pi_c_identity()

# death
transition_d = Transition(Compartment(x), Compartment(x + ContentChange(-1)), name='d')
k_d = Constant('k_d')
g_d = x[0] # TODO x should be enough here, in case D=1.
pi_d = pi_c_identity()

transitions = [
    (transition_I, k_I, g_I, pi_I),
    (transition_E, k_E, g_E, pi_E),
    (transition_b, k_b, g_b, pi_b),
    (transition_d, k_d, g_d, pi_d)
]
