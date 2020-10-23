from sympy import Eq, Symbol, Derivative, Basic
from compartor.compartments import Moment, Expectation, _getNumSpecies
from IPython.core.display import display


###################################################
#
# Displaying transitions in notebooks
#
###################################################

# -------------------------------------------------
def display_propensity_details(transition_class, name=None):
    tr, k, g, pi = transition_class.transition, transition_class.k, transition_class.g, transition_class.pi
    if name is None:
        name = tr.name
    if name is None:
        name = ''
    display(Eq(Symbol('k_{' + name + '}'), k, evaluate=False))
    display(Eq(Symbol('g_{' + name + '}'), g, evaluate=False))
    display(Eq(Symbol('\pi_{' + name + '}'), pi.expr, evaluate=False))


# -------------------------------------------------
def display_transition_classes(transitions):
    class Display(Basic):
        def __new__(cls, transitions):
            t = Basic.__new__(cls)
            t.transitions = transitions
            return t

        def __str__(self):
            return 'Display.__str__: TODO'

        def _sympystr(self, printer=None):
            return 'Display._sympystr: TODO'

        def _latex(self, printer=None):
            ll = []
            for t in self.transitions:
                tl = t.transition._latex(printer, True)
                pl = t._propensity_latex(printer)
                ll.append(r"%s && %s" % (tl, pl))
            return r"\begin{align} %s \end{align}" % r"\\".join(ll)
    display(Display(transitions))


###################################################
#
# Displaying expected moment evolution in notebooks
#
###################################################

# -------------------------------------------------
def display_moment_equation(expr_moment, expr_dfMdt, D=None):
    """
    :param expr_moment: lhs of evolution equation
    :param expr_dfMdt: rhs of evolution equation
    :param D: if present, Moment([0]*D) will be replaced by N
    """
    if D is None:
        D = _getNumSpecies(expr_moment)
    lhs = Derivative(Expectation(expr_moment), Symbol('t'))
    rhs = expr_dfMdt
    evolution = Eq(lhs, rhs, evaluate=False)
    if not D is None:
        evolution = evolution.subs(Moment(*([0]*D)),Symbol('N'))
    display(evolution)


# -------------------------------------------------
def display_moment_equations(equations, D=None):
    for (fM, dfMdt) in equations:
        display_moment_equation(fM, dfMdt, D)
