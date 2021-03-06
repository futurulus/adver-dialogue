import re

WORD_RE_STR = r"""
(?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
|
(?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
|
(?:[\w_]+)                     # Words without apostrophes or dashes.
|
(?:\.(?:\s*\.){1,})            # Ellipsis dots.
|
(?:\*{1,})                     # Asterisk runs.
|
(?:\S)                         # Everything else that isn't whitespace.
"""

WORD_RE = re.compile(r"(%s)" % WORD_RE_STR, re.VERBOSE | re.I | re.UNICODE)


def basic_unigram_tokenizer(s, lower=True):
    words = WORD_RE.findall(s)
    if lower:
        words = [w.lower() for w in words]
    return words


def basic_unigram_detokenizer(tokens, lower=True):
    tokens_spaced = []
    title = True
    for i, token in enumerate(tokens):
        if lower and title:
            token = token[:1].upper() + token[1:]
        punctuation = not token[0].isalnum()
        tokens_spaced.append(token if punctuation or i == 0 else ' ' + token)
        title = token.strip().endswith('.')
    return ''.join(tokens_spaced)


def whitespace_tokenizer(s, lower=True):
    if lower:
        s = s.lower()
    return s.split()


def whitespace_detokenizer(tokens, lower=True):
    tokens_capitalized = []
    title = True
    for i, token in enumerate(tokens):
        if lower and title:
            token = token[:1].upper() + token[1:]
        tokens_capitalized.append(token)
        title = token.strip().endswith('.')
    return ' '.join(tokens_capitalized)


TOKENIZERS = {
    'unigram': (basic_unigram_tokenizer, basic_unigram_detokenizer),
    'whitespace': (whitespace_tokenizer, whitespace_detokenizer),
}
