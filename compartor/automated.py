from compartor.closure import gamma_closures, substitute_closures, __getMomentPowers, meanfield_closures
from compartor.compartments import Moment, Expectation, compute_moment_equations, get_missing_moments, _getAndVerifyNumSpecies, _getNumSpecies

class AutomatedMomentClosureDetails():
    def __init__(self):
        self.desired = []
        self.added = []
        self.gamma = []
        self.meanfield = []

def automated_moment_equations(D, transition_classes, moments=None, display_details=True, details=None):
    """
    Outputs a closed system of moment equations for the provided transition classes.
    The moments to be characterized are identified automatically to comprise at least expected number and total mass dynamics.
    Optionally, a desired set of moments to be characterized can be passed as third argument.
    When necessary, third-order Gamma closures and possibly mean-field closures are applied.

    :param D: the number of chemical species
    :param transition_classes: a list of TransitionClass
    :param moments: optionally, a list of Moment expressions
    :return: moment equations as a list of pairs `(fM, dfMdt)`, where each pair consists of
        a moment expression and the derived expression for its time derivative.
    """
    if moments is None:
        indices = [0] * D
        moments = [Moment(*indices)]
        for d in range(D):
            indices[d] = 1
            moments.append(Moment(*indices))
            indices[d] = 0
    else:
        moments = moments.copy()
    _getAndVerifyNumSpecies(transition_classes, moments, D)
    desired = moments.copy()
    added = []
    while True:
        equations = compute_moment_equations(transition_classes, moments)
        missing = get_missing_moments(equations)
        gamma = []
        meanfield = []
        if missing:
            not_closed = []
            closures = gamma_closures(missing, not_closed)
            gamma = [a for a, _ in closures]
            worth_adding = _get_worth_adding(not_closed)
            if worth_adding:
                # Are there missing moments of order < 3?
                # Add them to the desired moments list and iterate
                added += worth_adding
                moments += worth_adding
                continue
            elif not_closed:
                # There are no missing moments of order < 3, however not all of them could be closed with gamma closure.
                # Add mean-field closures for those
                meanfield += not_closed
                closures += meanfield_closures(not_closed)
            # Now substitute the closures (gamma, possibly augmented by mean-field)
            equations = substitute_closures(equations, closures)
        # Either there were no missing moments or closures have been substituted, we're done.
        if not isinstance(details, AutomatedMomentClosureDetails):
            details = AutomatedMomentClosureDetails()
        details.desired = desired
        details.added = added
        details.gamma = gamma
        details.meanfield = meanfield
        if display_details:
            display_automated_moment_closure_details(details)
        return equations

def _worth_adding(expr):
    moment_powers = __getMomentPowers(expr)
    # sort by ascending pow
    moment_powers = sorted(moment_powers, key=lambda x: x[1])
    if len(moment_powers) < 3 and \
            sum([moment_powers[i][1] for i in range(len(moment_powers))]) < 3 and \
            sum([moment_powers[i][0].order() for i in range(len(moment_powers))]) < 3:
        return True
    else:
        return False

def _get_worth_adding(expressions):
    return [expr for expr in expressions if _worth_adding(expr)]

def display_automated_moment_closure_details(details):
    from IPython.core.display import display, Markdown
    from sympy import latex, Symbol
    def _print(expr):
        N = Moment(*([0] * _getNumSpecies(expr)))
        return f'${latex(Expectation(expr.subs(N, Symbol("N"))))}$'
    def _print_all(expressions):
        all_but_last = ", ".join([_print(expr) for expr in expressions[:-1]])
        last = _print(expressions[-1])
        if len(expressions) == 0:
            return ''
        elif len(expressions) == 1:
            return last
        elif len(expressions) == 2:
            return f'{all_but_last} and {last}'
        else:
            return f'{all_but_last}, and {last}'
    display(Markdown(f'Computed moment equations for desired moments {_print_all(details.desired)}.'))
    if details.added:
        display(Markdown(f'Equations were iteratively added for {_print_all(details.added)}.'))
    if details.gamma:
        display(Markdown(f'Gamma closures were substituted for {_print_all(details.gamma)}.'))
    if details.meanfield:
        display(Markdown(f'Mean-field closures were substituted for {_print_all(details.meanfield)}.'))
