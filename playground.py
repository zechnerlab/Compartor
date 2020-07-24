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

print(getContentPerSpecies(ContentVar('X') + ContentChange(0,-1,1), 3))


print("\n==========================\n")


gamma = IndexedBase('\gamma', integer=True, shape=1)
expr = (X+chng)**gamma
print(expr)

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
