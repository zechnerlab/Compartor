from sympy import Function


# -------------------------------------------------
class Moment(Function):
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
