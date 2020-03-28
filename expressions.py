import numpy as np
import random
import sympy
from util import run_with_timeout


tokens = ["x", "-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5",
          "+", "-", "*", "/",
          "exp", "log", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh",
          "atanh"]

valid_sympy_exps = [sympy.Symbol, sympy.Integer, sympy.Rational,
                    sympy.Add, sympy.Mul, sympy.exp, sympy.log, sympy.Pow, sympy.sin, sympy.cos, sympy.tan, sympy.asin,
                    sympy.acos, sympy.atan, sympy.sinh, sympy.cosh, sympy.tanh, sympy.asinh, sympy.acosh, sympy.atanh]


leaf_start = 0
binary_start = 11
unary_start = 15

leaf_values = range(leaf_start, binary_start)
binary_operators = range(binary_start, unary_start)
unary_operators = range(unary_start, len(tokens))

LL = len(leaf_values)
p_1 = len(unary_operators)
p_2 = len(unary_operators)


def is_leaf(x): return x < binary_start
def is_binary(x): return binary_start <= x < unary_start
def is_unary(x): return unary_start <= x


def compute_D(max_n):
    """
    Creates a matrix containing the D(e,n) values, representing the number of trees that can be generated from e empty
    elements and n internal nodes. See Appendix C in the paper.
    """
    max_e = max_n  # TODO: think about that
    # The 'e' dimension needs additional max_n space, because we have the recursive step D(e+1, n-1)
    D = -np.ones([max_e + max_n+1, max_n+1])  # Initialize with -1
    D[0, :] = 0
    for e in range(D.shape[0]):
        D[e, 0] = 1

    # Use dynamic programming for a simple O(n^2) solution
    def get_value(e, n):
        if D[e, n] == -1:
            D[e, n] = get_value(e-1, n) + get_value(e, n-1) + get_value(e+1, n-1)
        return D[e, n]

    for e in range(max_e + 1):
        for n in range(max_n + 1):
            D[e, n] = get_value(e, n)
    return D


def compute_L(max_n, bins):
    """
    Creates a matrices L of shape [e, n, bins] representing the probability distribution of the position and arity
    (k, a) of the next unary and binary node to allocate. See Appendix C in the paper. Not to be confused with LL, the
    number of possible leaf values (in the paper both are called L).

    To represent the distribution, the possible position and arity tuples are distributed across a number of bins accor-
    ding to the distribution. A higher value of 'bins' therefore increases the "resolution" of the distribution. The
    distribution can be sampled by generating a random number r between 0 and bins-1 and accessing the tuple L[e, n, r].
    """
    max_e = max_n  # TODO: think about that
    D = compute_D(max_n)
    L = np.zeros([D.shape[0], D.shape[1], bins], dtype=(int, 2))
    for e in range(1, max_e + 1):
        for n in range(max_n + 1):
            # Save the index of the next bin to allocate
            index = 0
            # For each possible position k and arity a
            for k in range(e):
                for a in [1, 2]:
                    # Calculate probability for L(e,n) = (k,a)
                    if a == 1:
                        prob = D[e-k, n-1] / float(D[e, n])
                    else:
                        prob = D[e-k+1, n-1] / float(D[e, n])
                    # How many bins should (k, a) get?
                    b = int(prob*bins + 0.5)
                    # Allocate the bins
                    L[e, n, index:index + b] = (k, a)
                    index += b
    return L


def expression_generator(max_n):
    """ Generates expression with <= max_n internal nodes in the form of prefix expression lists. """
    bins = 1000 * max_n
    L = compute_L(max_n, bins=bins)
    while True:
        exp = []
        e = 1
        for n in range(max_n, 0, -1):
            k, a = L[e, n, random.randrange(0, bins)]
            # Add k leaves
            exp += [random.choice(leaf_values) for _ in range(k)]
            if a == 1:
                # Unary operator
                exp.append(random.choice(unary_operators))
                e = e - k
            else:
                # Binary operator
                exp.append(random.choice(binary_operators))
                e = e - k + 1
        # All remaining empty nodes become leaves
        exp += [random.choice(leaf_values) for _ in range(e)]
        yield exp


def infix(exp):
    """ Returns an infix string representation giving a prefix token list. """
    stack = []
    for x in reversed(exp):
        if is_leaf(x):
            stack.append(tokens[x])
        elif is_unary(x):
            arg = stack.pop()
            stack.append(tokens[x] + "(" + arg + ")")
        else:
            right = stack.pop()
            left = stack.pop()
            stack.append("(" + left + " " + tokens[x] + " " + right + ")")
    assert len(stack) == 1
    return stack[0]


def prefix_to_sympy(exp):
    """
    Converts a prefix expression list into a sympy expression. Expressions will automatically be simplified. If an
    expression is invalid as described in the paper, `None` will be returned.
    """
    exp = run_with_timeout(sympy.sympify, (infix(exp),), 0.25)
    if exp is None:
        return None
    # Walk over the tree and check if all sub expressions are valid
    for e in sympy.preorder_traversal(exp):
        if all(not isinstance(e, v) for v in valid_sympy_exps):
            return None
    return exp

