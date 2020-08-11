from compartmentsB import *

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
def __getMoments(expr):
    """
    Get all instances of Moment(...) occurring in expr
    :param expr:
    :return:
    """
    if expr.func == Add or expr.func == Mul:
        moments = [__getMoments(arg) for arg in expr.args]
        return list(set(itertools.chain(*moments)))
    elif expr.func == Pow:
        return __getMoments(expr.args[0])
    elif expr.func == Moment:
        return [expr]
    elif issubclass(expr.func, Integer):
        return []
    else:
        raise TypeError("Unexpected expression " + str(expr))


# -------------------------------------------------
def __ito(expr):
    """
    Apply Ito's rule to a function of moments.
    :param expr: expression comprising Moments, addition, multiplication, and powers with moments only in the base
    :return: expansion by Ito's rule
    """
    moments = __getMoments(expr)
    substitutions = [(m, m + DeltaM(*m.args)) for m in moments]
    expr = expr.subs(substitutions) - expr
    return expr.expand()





# -------------------------------------------------

# temporary export for playground

def ito(expr):
    return __ito(expr)


