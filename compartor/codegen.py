from sympy import Add, Mul, Pow, Number, Symbol, simplify
from compartor.compartments import Moment, Expectation

import itertools
import collections

# -------------------------------------------------
# helpers
# -------------------------------------------------
def _get_constants(expr):
    if isinstance(expr, collections.abc.Iterable): # assuming it is an iterable of (fM, dfMdt) tuples
        return set(itertools.chain(*(_get_constants(dfMdt) for _, dfMdt in expr)))
    elif expr.func in [Add, Mul, Pow, Expectation]:
        return set(itertools.chain(*(_get_constants(arg) for arg in expr.args)))
    elif expr.func == Moment or issubclass(expr.func, Number):
        return {}
    else:
        return {expr}


def _fuzzy_expr_lookup(dictionary, expr):
    """
    Get the value in the dictionary with a key that matches expr in the following way:
    If expr respectively key are sympy Symbols, extract the symbol name for comparison.
    Otherwise assume that expr respectively key are strings and use directly for comparison.

    :param dictionary:
    :param expr: a Symbol or a symbol name
    :return:
    """
    def name(k):
        return k.name if issubclass(type(k), Symbol) else k
    expr_name = name(expr)
    for k, v in dictionary.items():
        if name(k) == expr_name:
            return v
    raise KeyError(expr)


def _gen_comment_text(expr, comment_species_base_index = 1):
    """
    Generate text describing a given moments expression for putting into comments in generated code.
    For example, expr=Moment(0) will produce "N (Number of Compartments)", and expr=Moment(1,0) will produce
    "M1 (Total Mass of species 1)".

    :param expr: moments expression
    :param comment_species_base_index: base index for species to use in comments.
        E.g., if setting comment_species_base_index=0, then expr=Moment(1,0) will produce "M0 (Total Mass of species 0)".
    :return:
    """
    def is_N(moment):
        return sum(moment.args) == 0

    def is_M(moment):
        return sum(moment.args) == 1

    def is_M_squ(moment):
        return is_single_species(moment) and sum(moment.args) == 2

    def is_single_species(moment):
        return sum(1 for pow in moment.args if pow)

    def species(moment):
        return next((species + comment_species_base_index, pow) for species, pow in enumerate(moment.args) if pow)

    def moment_type_and_species(moment):
        if is_N(moment):
            return ('N', 0)
        elif is_M(moment):
            i, _ = species(moment)
            return ('M', i)
        elif is_M_squ(moment):
            i, _ = species(moment)
            return ('S', i)
        else:
            return (None, 0)

    def moment_name(expr):
        if expr.func is Moment:
            D = len(expr.args)
            type, i = moment_type_and_species(expr)
            if D == 1:
                i = ''
            if type == 'N':
                return 'N'
            elif type == 'M':
                return f'M{i}'
            elif type == 'S':
                return f'S{i}'
        elif expr.func is Pow:
            name = moment_name(expr.args[0])
            if name:
                return f'{name}^{expr.args[1]}'
        elif expr.func is Mul:
            factors = [moment_name(arg) for arg in expr.args]
            for name in factors:
                if name is None:
                    return None
            else:
                return '*'.join(factors)
        return None

    if expr.func is Moment:
        D = len(expr.args)
        name, i = moment_type_and_species(expr)
        if name == 'N':
            return 'Number of Compartments (N)'
        elif name == 'M':
            if D == 1:
                return 'Total Mass'
            else:
                return f'Total Mass of species {i}'
        elif name == 'S':
            if D == 1:
                return 'Sum of squared content'
            else:
                return f'Sum of squared content of species {i}'
    else:
        return moment_name(expr)


# -------------------------------------------------
# expression tree
# -------------------------------------------------
class _expr_atom:
    def __init__(self, arg):
        self.arg = arg

    def code(self):
        return f'{self.arg}'

class _expr_add:
    def __init__(self, *args):
        self.args = args

    def code(self):
        code = ''
        for arg in self.args:
            c = arg.code()
            if not code:
                code = c
            else:
                if c.startswith('-'):
                    code += c
                else:
                    code += '+' + c
        return code

class _expr_mul:
    def __init__(self, *args):
        self.args = args

    def code(self):
        def paren(arg):
            c = arg.code()
            if type(arg) is _expr_add:
                return f'({c})'
            else:
                return c
        return '*'.join([paren(arg) for arg in self.args])

class _expr_div:
    def __init__(self, dividend, divisor):
        self.dividend = dividend
        self.divisor = divisor

    def code(self):
        c1 = self.dividend.code()
        if type(self.dividend) is _expr_add:
            c1 = f'({c1})'
        c2 = self.divisor.code()
        if c2.startswith('-') or type(self.divisor) in [_expr_add, _expr_mul]:
            c2 = f'({c2})'
        return f'{c1}/{c2}'

class _expr_pow:
    def __init__(self, base, exponent, generator):
        self.base = base
        self.exponent = exponent
        self.generator = generator

    def code(self):
        def paren(arg):
            c = arg.code()
            if c.startswith('-') or type(arg) in [_expr_add, _expr_mul, _expr_div]:
                return f'({c})'
            else:
                return c
        return self.generator.format_pow(paren(self.base), paren(self.exponent))


# -------------------------------------------------
class AbstractCodeGenerator:
    def __init__(self):
        self._code = ''
        self._indent = 0
        self.cr = '\n'

        # the base for array indices (e.g., 0 for python, 1 for julia)
        self.index_base = 0

        # dictionary mapping moment expression to index i in M[i], dM[i], M0[i] variables in the generated code
        # Note that this needs not necessarily be used, depending on how gen_M, gen_dM, gen_M0 functions are overridden
        self.moment_indices = None

        # dictionary mapping constants (in moment expressions) to variables in the generated code
        # Note that this needs not necessarily be used, depending on how the gen_constant function is overridden
        self.constant_variables = None

        # dictionary mapping constants (in moment expressions) to expressions (in the generated code)
        # for initializing the corresponding variables
        self.constant_initializers = None

        # dictionary mapping moments to expressions (in the generated code) for initializing the corresponding variables
        self.moment_initializers = None

        # whether the generated code should include comments above dM[i] = ... expressions
        # about which moment M[i] corresponds to
        self.gen_moment_comments = True

    def gen_M(self, expr):
        """
        Generate variable name for a given Moment.
        Can be overridden in derived code generators.

        By default, this produces "M[i]", i being the index assigned by self.moment_indices.
        """
        return f'M[{self._moment_indices[expr]}]'

    def gen_dM(self, expr):
        """
        Generate variable name for a given Moment derivative.
        Can be overridden in derived code generators.

        By default, this produces "dM[i]", i being the index assigned by self.moment_indices.
        """
        return f'dM[{self._moment_indices[expr]}]'

    def gen_constant(self, expr):
        """
        Generate variable name for a given constant.
        Can be overridden in derived code generators.

        By default, this just looks up expr in the self.constant_variables dictionary.
        """
        return _fuzzy_expr_lookup(self._constant_variables, expr)

    def gen_constant_init(self, expr):
        """
        Generate initialization expression for a given constant.
        Can be overridden in derived code generators.

        By default, this just looks up expr in the self.constant_initializers dictionary.
        """
        return _fuzzy_expr_lookup(self._constant_initializers, expr)

    def indent(self, amount):
        self._indent += amount

    def append_statement(self, line):
        self._code += ' ' * self._indent + line + self.cr

    def append_comment(self, line):
        self._code += ' ' * self._indent + self.format_comment(line) + self.cr

    def __str__(self):
        return self._code

    def __gen_mul(self, expr):
        """
        Helper for generating code _expr for Mul.
        Takes into account the sign on exponents and rearranges into a _expr_div if the are negative exponents.
        """
        def to_divisor(i):
            if i.func == Pow and issubclass(i.args[1].func, Number) and i.args[1] < 0:
                return Pow(i.args[0], -i.args[1])
            else:
                return None
        pos = [i for i in expr.args if to_divisor(i) is None]
        neg = [to_divisor(i) for i in expr.args if to_divisor(i)]
        if not pos:
            pos = [1]
        def mul(args):
            cf = [self._gen_code_expr(i) for i in args]
            if len(cf) == 1:
                return cf[0]
            else:
                return _expr_mul(*cf)
        if not neg:
            return mul(pos)
        else:
            return _expr_div(mul(pos), mul(neg))

    def _gen_code_expr(self, expr):
        """
        Generate code for the specified expression.
        This will return one of the the _expr_xxx classes (_expr_atom, _expr_add, etc).
        Use result.code() to generate a string from that.
        """
        if expr.func == Add:
            return _expr_add(*[self._gen_code_expr(i) for i in expr.args])
        elif expr.func == Mul:
            return self.__gen_mul(expr)
            # return _expr_mul(*[self._gen_code_expr(i) for i in expr.args])
        elif expr.func == Pow:
            return _expr_pow(self._gen_code_expr(expr.args[0]), self._gen_code_expr(expr.args[1]), self)
        elif expr.func == Moment:
            return _expr_atom(self.gen_M(expr))
        elif expr.func == Expectation:
            return _expr_atom(self.gen_M(expr.args[0]))
        elif issubclass(expr.func, Number):
            return _expr_atom(expr)
        else:
            return _expr_atom(self.gen_constant(expr))

    def _initialization_order(self, index_map):
        """
        Order moment expressions in index_map such that
        1.) dependencies come later in the order than all their dependants and
        2.) moment expressions that are not ordered by (1) are ordered by ascending index
        :param index_map: maps moment expression to index in M vector of generated code
        :return: ordered list of moment expression
        """
        order = []

        def dependencies(expr):
            if expr.func == Add or expr.func == Mul:
                return set(itertools.chain(*(dependencies(arg) for arg in expr.args)))
            elif expr.func == Pow:
                return dependencies(expr.args[0])
            elif expr.func == Moment:
                return {expr}
            elif issubclass(expr.func, Number):
                return {}
            else:
                raise TypeError("Unexpected expression " + str(expr))

        deps = []
        for expr, i in index_map.items():
            d = dependencies(expr).difference({expr})
            deps.append((expr, d, i))

        def sortkey(tuple):
            expr, d, i = tuple
            return 9999 if d else i

        while deps:
            deps.sort(key=sortkey)
            expr, d, i = deps.pop(0)
            if d:
                raise RuntimeError(f'depenency missing: {d}')
            order.append(expr)
            deps = [(expr1, d1.difference({expr}), i1) for expr1, d1, i1 in deps]

        return order

    def _default_moment_indices(self, equations):
        moments = [fM for fM, dfMdt in equations]
        return {moment: i + self.index_base for i, moment in enumerate(moments)}

    def _default_constant_variables(self, equations):
        constants = _get_constants(equations)
        return {constant: f'c{i}' for i, constant in enumerate(constants)}

    def _default_constant_initializers(self, equations):
        constants = _get_constants(equations)
        return {constant: '??? ' + self.format_comment(f'{constant}, please specify!') for constant in constants}

    def _default_moment_initializers(self, equations):
        moments = [fM for fM, dfMdt in equations if fM.func is Moment]
        return {moment: '??? ' + self.format_comment(f'initial value for {moment}, please specify!') for moment in moments}

    def _init_dictionaries(self, equations):
        if self.moment_indices is None:
            self._moment_indices = self._default_moment_indices(equations)
        else:
            self._moment_indices = self.moment_indices

        if self.constant_variables is None:
            self._constant_variables = self._default_constant_variables(equations)
        else:
            self._constant_variables = self.constant_variables

        if self.constant_initializers is None:
            self._constant_initializers = self._default_constant_initializers(equations)
        else:
            self._constant_initializers = self.constant_initializers

        if self.moment_initializers is None:
            self._moment_initializers = self._default_moment_initializers(equations)
        else:
            self._moment_initializers = self.moment_initializers

    def gen_ODEs_body(self, equations):
        for k, v in self._constant_initializers.items():
            self.append_statement(f'{self.gen_constant(k)} = {v}')

        for fM, dfMdt in equations:
            # c = gen_comment_text(fM)
            # self.append_statement(f'# {"???" if c is None else c}')
            c = self._gen_code_expr(simplify(dfMdt)).code()
            c = c.replace("-1*", "-")
            comment = _gen_comment_text(fM)
            if self.gen_moment_comments and comment:
                self.append_comment(comment)
            self.append_statement(f'{self.gen_dM(fM)} = {c}')

    def gen_initial_body(self, equations):
        index_map = {fM : i for i, (fM, dfMdt) in enumerate(equations)}
        exprs = self._initialization_order(index_map)
        for expr in exprs:
            comment = _gen_comment_text(expr)
            if self.gen_moment_comments and comment:
                self.append_comment(comment)
            if expr.func is Moment:
                self.append_statement(f'{self.gen_M(expr)} = {self._moment_initializers[expr]}')
            else:
                self.append_statement(f'{self.gen_M(expr)} = {self._gen_code_expr(expr).code()}')


# -------------------------------------------------
class GenerateJulia(AbstractCodeGenerator):
    def __init__(self):
        super().__init__()
        self.index_base = 1

    def format_comment(self, text):
        return f'# {text}'

    def format_pow(self, base, exp):
        return f'{base}^{exp}'

    def generate(self, equations, function_name = "generated"):
        self._init_dictionaries(equations)

        self.append_comment("evaluate ODEs")
        self.append_statement(f'function {function_name}_ODEs(dM, M, parameters, t)')
        self.indent(2)
        self.gen_ODEs_body(equations)
        self.append_statement("return")
        self.indent(-2)
        self.append_statement("end")
        self.append_statement("")

        self.append_comment("initialize expected moments vector")
        self.append_statement(f'function {function_name}_initial(n0)')
        self.indent(2)
        self.append_statement(f'M=zeros[{len(equations)}]')
        self.gen_initial_body(equations)
        self.append_statement("return M")
        self.indent(-2)
        self.append_statement("end")
        self.append_statement("")
        return self._code


# -------------------------------------------------
def generate_julia_code(equations, function_name = "generated"):
    generator = GenerateJulia()
    code = generator.generate(equations, function_name=function_name)
    return code


# -------------------------------------------------
class GeneratePython(AbstractCodeGenerator):
    def __init__(self):
        super().__init__()
        self.index_base = 0

        # a list of strings or symbols that specifies in which order constants appear in the parameters[] argument to
        # the generated ..._ODEs() function
        self.constants_parameter_order = None

    def format_comment(self, text):
        return f'# {text}'

    def format_pow(self, base, exp):
        return f'{base}**{exp}'

    def _default_moment_initializers(self, equations):
        moments = [fM for fM, dfMdt in equations if fM.func is Moment]
        return {moment: f'initial[{i}]' for i, moment in enumerate(moments)}

    def _default_constant_initializers(self, equations):
        if self.constants_parameter_order:
            # if user set a constants_order, we use that to determine indices in parameters[] array
            def constants_order(expr):
                def name(k):
                    return k.name if issubclass(type(k), Symbol) else k
                expr_name = name(expr)
                return next(i for i, c in enumerate(self.constants_parameter_order) if name(c) == expr_name)
            key = constants_order
        else:
            # otherwise, we use lexicographical order of constant names
            key = lambda x: str(x)
        constants = sorted(_get_constants(equations), key=key)
        return collections.OrderedDict(
            (constant, f'parameters[{i}] ' + self.format_comment(f'{constant}'))
            for i, constant in enumerate(constants))

    def generate(self, equations, function_name = "generated"):
        self._init_dictionaries(equations)

        self.append_comment("evaluate ODEs")
        self.append_statement(f'def {function_name}_ODEs(M, dM, parameters):')
        self.indent(2)
        self.append_statement('"""')
        self.append_statement('Evaluate derivatives of expected moments')
        self.append_statement('')
        self.append_statement('Indices in M and dM vectors are ')
        ordered = sorted(self._moment_indices.items(), key=lambda t: t[1])
        for m, i in ordered:
            self.append_statement(f'  M[{i}] = {m}')
        self.append_statement('')
        self.append_statement(':param M: expected moments')
        self.append_statement(':param dM: result, the derivative dM/dt is stored here')
        self.append_statement(':param parameters: tuple of values for constants (' +
                              ', '.join([f'{constant}' for constant in self._constant_initializers]) +
                              ')')
        self.append_statement(':return: dM')
        self.append_statement('"""')
        self.gen_ODEs_body(equations)

        self.append_statement("return dM")
        self.indent(-2)
        self.append_statement("")

        self.append_comment("initialize expected moments vector")
        self.append_statement(f'def {function_name}_initial(initial):')
        self.indent(2)
        self.append_statement('"""')
        self.append_statement('Create inital expected moments vector')
        self.append_statement('')
        if not self.moment_initializers:
            moments = [fM for fM, dfMdt in equations if fM.func is Moment]
            self.append_statement(':param initial: tuple of initial values for expectations of (' +
                                  ', '.join([f'{moment}' for moment in moments]) +
                                  ')')
        self.append_statement(':return: initial expected moments vector')
        self.append_statement('"""')
        self.append_statement('import numpy as np')
        self.append_statement(f'M = np.zeros({len(equations)})')
        self.gen_initial_body(equations)
        self.append_statement("return M")
        self.indent(-2)
        self.append_statement("")
        return self._code


# -------------------------------------------------
def generate_python_code(equations, function_name = "generated"):
    generator = GeneratePython()
    code = generator.generate(equations, function_name=function_name)
    return code
