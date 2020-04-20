special = ["<pad>", "<sos>", "<eos>"]  # Padding, start of sentence, end of sentence

constants = ["pi", "E"]

variables = ["x", "a", "b", "c"]

operators = ["add", "sub", "mul", "div", "pow", "rac", "inv", "pow2", "pow3", "pow4", "pow5", "sqrt", "exp", "ln",
             "abs", "sign", "sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "acot", "asec", "acsc",
             "sinh", "cosh", "tanh", "coth", "sech", "csch", "asinh", "acosh", "atanh", "acoth", "asech", "acsch"]

symbols = ["I", "INT+", "INT-"] + [str(i) for i in range(10)]

synonyms = {"log": "ln", "+": "add", "-": "sub", "*": "mul", "/": "div", "e": "E", "i": "I"}


vocab = special + constants + variables + operators + symbols
_token2id = {vocab[i]: i for i in range(len(vocab))}
_id2token = {i: vocab[i] for i in range(len(vocab))}


def token2id(t):
    try:
        return _token2id[t]
    except KeyError:
        return _token2id[synonyms[t]]


def id2token(i):
    return _id2token[i]

