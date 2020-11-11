from sympy import Eq, Symbol, Derivative, Basic
from compartor.compartments import Moment, Expectation, _getNumSpecies, get_missing_moments
from IPython.core.display import display, Markdown


###################################################
#
# Displaying reaction networks and moment equations
# in jupyter notebooks
#
###################################################

# -------------------------------------------------
def display_propensity_details(transition_class, name=None):
    k, g, pi = transition_class.k, transition_class.g, transition_class.pi
    if name is None:
        name = transition_class.name
    if name is None:
        name = ''
    display(Eq(Symbol('k_{' + name + '}'), k, evaluate=False))
    display(Eq(Symbol('g_{' + name + '}'), g, evaluate=False))
    display(Eq(Symbol('\pi_{' + name + '}'), pi.expr, evaluate=False))


# -------------------------------------------------
def display_transition_classes(transitions):
    class Display(Basic):
        def __new__(cls, transitions):
            obj = Basic.__new__(cls)
            obj.transitions = transitions
            return obj

        def __str__(self):
            return 'Display.__str__: TODO'

        def _sympystr(self, printer=None):
            return 'Display._sympystr: TODO'

        def _latex(self, printer=None):
            ll = []
            for t in self.transitions:
                tl = t.transition._latex(printer, align=True, name=t.name)
                pl = t._propensity_latex(printer, name=t.name)
                ll.append(r"%s && %s" % (tl, pl))
            return r"\begin{align} %s \end{align}" % r"\\".join(ll)
    display(Display(transitions))


# -------------------------------------------------
def display_moment_equation(expr_moment, expr_dfMdt):
    """
    :param expr_moment: lhs of evolution equation
    :param expr_dfMdt: rhs of evolution equation
    """
    D = _getNumSpecies(expr_moment)
    lhs = Derivative(Expectation(expr_moment), Symbol('t'))
    rhs = expr_dfMdt
    evolution = Eq(lhs, rhs, evaluate=False)
    if not D is None:
        evolution = evolution.subs(Moment(*([0]*D)),Symbol('N'))
    display(evolution)


# -------------------------------------------------
def display_moment_equations(equations, print_missing=True):
    for (fM, dfMdt) in equations:
        display_moment_equation(fM, dfMdt)
    if print_missing:
        missing = get_missing_moments(equations)
        if missing:
            display(Markdown('**The system is not closed!** Moment equations are missing for:'))
            display(missing)


# -------------------------------------------------
def display_closures(closures):
    for m, c in closures:
        D = _getNumSpecies(m)
        display(Eq(Expectation(m), c, evaluate=False).subs(Moment(*([0] * D)), Symbol('N')))
