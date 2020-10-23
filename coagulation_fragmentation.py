from sympy import Symbol
from compartor import Content, Transition, TransitionClass, EmptySet, Compartment, Constant, OutcomeDistribution

D = 1 # number of species

x = Content('x')
y = Content('y')

# Intake
transition_I = Transition(EmptySet, Compartment(y), name='I')
k_I = Constant('k_I')
g_I = 1
pi_I = OutcomeDistribution.poisson(
    Symbol("\pi_{Poiss}(y; \lambda)"),
    y[0],
    Symbol("\lambda"))
Intake = TransitionClass(transition_I, k_I, g_I, pi_I)

# Exit
transition_E = Transition(Compartment(x), EmptySet, name='E')
k_E = Constant('k_E')
g_E = 1
Exit = TransitionClass(transition_E, k_E, g_E)

# Coagulation
transition_C = Transition(Compartment(x) + Compartment(y), Compartment(x + y), name='C')
k_C = Constant('k_C')
g_C = 1
Coagulation = TransitionClass(transition_C, k_C, g_C)

# Fragmentation
transition_F = Transition(Compartment(x), Compartment(y) + Compartment(x-y), name='F')
k_F = Constant('k_F')
g_F = x[0]
pi_F = OutcomeDistribution.uniform(Symbol("\pi_F(y|x)"), y[0], 0, x[0])
Fragmentation =TransitionClass(transition_F, k_F, g_F, pi_F)

transitions = [Intake, Exit, Coagulation, Fragmentation]
