from sympy import *
import collections
import itertools

from compartmentsB import *

# --- main ---

X = ContentVar('X')
Y = ContentVar('Y')

checkSimpleCompartment(Compartment(X))

w = getSumMassAction({Compartment(X):1, Compartment(Y):1}, 1)
print(w)

w = getSumMassAction({Compartment(X):1})
print(w)


print("\n==========================\n")


X = ContentVar('X')
chng = ContentChange(0,-1,1)

content_per_species = getContentPerSpecies(X + chng, 3)
print(content_per_species)


print("\n==========================\n")


gamma = IndexedBase('\gamma', integer=True, shape=1)

expr = mpow(content_per_species, gamma)
print(expr)

expr = expr.subs({gamma[0]:0, gamma[1]:2, gamma[2]:0})
print(expr)

expr = expr.expand()
print(expr)


print("\n==========================\n")


X = ContentVar('X')
Y = ContentVar('Y')

expr = deltaM({Compartment(X):-1, Compartment(X+chng):1}, 3)
print(expr)

expr = substituteGamma(expr, 0, 1, 1)
print(expr)

expr = expr.expand()
print(expr)


print("\n==========================\n")


print_latex(Moment(0,1))

# C = Context(2)
#
# Exit = Transition(C.compartment(X), EmptySet())
# kExit = symbols('k_{Exit}')
# gExit = 1
# piExit = 1
#
# print(Exit)
#
# C.doit(Exit, (kExit, gExit, piExit))
