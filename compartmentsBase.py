from sympy import Function, IndexedBase, Basic, Symbol


# -------------------------------------------------
class Moment(Function):
    """
    Expression for M^\gamma, args are the elements of \gamma
    """
    def __str__(self):
        return f'Moment{self.args}'

    def _latex(self, printer=None, exp=1):
        b = self.__base_latex(printer=printer)
        if exp == 1:
            return b
        else:
            return '{\\left(' + b + '\\right)}^{' + printer.doprint(exp) + '}'

    def __base_latex(self, printer=None):
        if len(self.args) == 0:
            return 'M^{\gamma}'
        elif len(self.args) == 1:
            return 'M^{' + printer.doprint(self.args[0]) + '}'
        else:
            return 'M^{\\left(' + ", ".join([printer.doprint(arg) for arg in self.args]) + '\\right)}'


# -------------------------------------------------
class DeltaM(Function):
    """
    Expression for \Delta{}M^\gamma, args are the elements of \gamma
    """
    def __str__(self):
        return f'DeltaM{self.args}'

    def _latex(self, printer=None, exp=1):
        b = self.__base_latex(printer=printer)
        if exp == 1:
            return b
        else:
            return '{\\left(' + b + '\\right)}^{' + printer.doprint(exp) + '}'

    def __base_latex(self, printer=None):
        if len(self.args) == 0:
            return '\Delta{}M^{\gamma}'
        elif len(self.args) == 1:
            return '\Delta{}M^{' + printer.doprint(self.args[0]) + '}'
        else:
            return '\Delta{}M^{\\left(' + ", ".join([printer.doprint(arg) for arg in self.args]) + '\\right)}'


# -------------------------------------------------
class Expectation(Function):
    """
    TODO
     This is just used for printing now.
     Look into sympy.stats for "real" random variables, expectation, etc.
     https://docs.sympy.org/latest/modules/stats.html
     Maybe can define probability space over n and actually use it...?
    """
    nargs = 1

    def __str__(self):
        return f'E[{self.args[0]}]'

    def _sympystr(self, printer=None):
        return f'E[{self.args[0]}]'

    def _latex(self, printer=None):
        return '\\left< ' + printer.doprint(self.args[0]) + '\\right> '


###################################################
#
# Specifying transitions and propensities
#
###################################################


# -------------------------------------------------
def ContentVar(name):
    """
    Create a content variable.
    If X is a ContentVar, then X[i] refers to content for species i.
    """
    return IndexedBase(name, integer=True, shape=1)


# -------------------------------------------------
class ContentChange(Function):
    """
    An integer vector that can be added to ContentVar to express chemical modifications.
    args are change per species.
    """
    def __str__(self):
        return f'{self.args}'

    def _sympystr(self, printer=None):
        return f'{self.args}'

    def _latex(self, printer=None):
        return printer.doprint(self.args)


# -------------------------------------------------
class Compartment(Function):
    """
    Expression for a compartment, with one argument that is the expression for the compartment content.
    """
    nargs = 1

    def __str__(self):
        return f'[{self.args[0]}]'

    def _sympystr(self, printer=None):
        return f'[{self.args[0]}]'

    def _latex(self, printer=None):
        return '\\left[' + printer.doprint(self.args[0]) + '\\right]'

    def content(self):
        return self.args[0]


# -------------------------------------------------
class Transition(Basic):
    """
    Expression for a transition with lhs and rhs specifying sums of compartments
    """
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f'{self.lhs} ---> {self.rhs})'

    def _latex(self, printer=None):
        # Always use printer.doprint() otherwise nested expressions won't
        # work. See the example of ModOpWrong.
        l = printer.doprint(self.lhs)
        r = printer.doprint(self.rhs)
        return l + '\longrightarrow{}' + r



# -------------------------------------------------
def Constant(name):
    return Symbol(name, real=True, constant=True)


# -------------------------------------------------
__numCompartments = Function('n', integer=True)

def n(content):
    """
    Expression for number of compartments with given content
    """
    if content.func == Compartment:
        return n(content.args[0])
    return __numCompartments(content)



