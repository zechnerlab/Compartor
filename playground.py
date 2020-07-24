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
