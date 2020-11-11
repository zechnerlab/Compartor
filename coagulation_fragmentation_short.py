from sympy import Symbol
from compartor import Content, TransitionClass, OutcomeDistribution, to

x = Content('x')
y = Content('y')

pi_I = OutcomeDistribution.poisson(Symbol("\pi_{Poiss}(y; \lambda)"), y[0], Symbol("\lambda"))
pi_F = OutcomeDistribution.uniform(Symbol("\pi_F(y|x)"), y[0], 0, x[0])

Intake        = TransitionClass( {}        -to> [y],         'k_I',         pi=pi_I, name='I')
Exit          = TransitionClass( [x]       -to> {},          'k_E',                  name='E')
Coagulation   = TransitionClass( [x] + [y] -to> [x+y],       'k_C',                  name='C')
Fragmentation = TransitionClass( [x]       -to> [y] + [x-y], 'k_F', g=x[0], pi=pi_F, name='F')

transitions = [Intake, Exit, Coagulation, Fragmentation]
