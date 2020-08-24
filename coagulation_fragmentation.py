from sympy import Symbol
from compartments import ContentVar, Transition, EmptySet, Compartment, Constant, pi_c_poisson, pi_c_identity, pi_c_uniform

D = 1 # number of species

x = ContentVar('x')
y = ContentVar('y')

# Intake
transition_I = Transition(EmptySet(), Compartment(y), name='I')
k_I = Constant('k_I')
g_I = 1
pi_I = pi_c_poisson(
    Symbol("\pi_I", positive=True),
    y[0],
    Symbol("\lambda", positive=True))

# Exit
transition_E = Transition(Compartment(x), EmptySet(), name='E')
k_E = Constant('k_E')
g_E = 1
pi_E = pi_c_identity()

# Coagulation
transition_C = Transition(Compartment(x) + Compartment(y), Compartment(x + y), name='C')
k_C = Constant('k_C')
g_C = 1
pi_C = pi_c_identity()

# Fragmentation
transition_F = Transition(Compartment(x), Compartment(y) + Compartment(x-y), name='F')
k_F = Constant('k_F')
g_F = x[0]
pi_F = pi_c_uniform(Symbol("\pi_F", positive=True), y[0], 0, x[0])

transitions = [
    (transition_I, k_I, g_I, pi_I),
    (transition_E, k_E, g_E, pi_E),
    (transition_C, k_C, g_C, pi_C),
    (transition_F, k_F, g_F, pi_F)
]
