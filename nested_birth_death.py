from sympy import Symbol
from compartments import Content, Transition, TransitionClass, EmptySet, Compartment, Constant, ContentChange, OutcomeDistribution

D = 1 # number of species

y = Content('y')
x = Content('x')

# Intake
transition_I = Transition(EmptySet, Compartment(y), name='I')
k_I = Constant('k_I')
g_I = 1
pi_I = OutcomeDistribution.poisson(
    Symbol("\pi_{Poiss}(y; \lambda)", positive=True),
    y[0],
    Symbol("\lambda", positive=True))
Intake = TransitionClass(transition_I, k_I, g_I, pi_I)

# Exit
transition_E = Transition(Compartment(x), EmptySet, name='E')
k_E = Constant('k_E')
g_E = 1
Exit = TransitionClass(transition_E, k_E, g_E)

# birth
transition_b = Transition(Compartment(x), Compartment(x + ContentChange(1)), name='b')
k_b = Constant('k_b')
g_b = 1
Birth = TransitionClass(transition_b, k_b, g_b)

# death
transition_d = Transition(Compartment(x), Compartment(x + ContentChange(-1)), name='d')
k_d = Constant('k_d')
g_d = x[0]
Death = TransitionClass(transition_d, k_d, g_d)

transitions = [Intake, Exit, Birth, Death]
