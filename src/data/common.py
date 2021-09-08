from typing import List, TypeVar, Iterable, Sequence
import re
import itertools

PREM_KEY = 'premise'
HYPO_KEY = 'hypothesis'
LABEL_KEY = 'label'
SENT_KEY = 'sentence'
ANTI_KEY = 'neg_sentence'
MASKED_SENT_KEY = 'masked_sentence'
MASKED_ANTI_KEY = 'masked_neg_sentence'


def contoken(num: int) -> str:
    return "<C{}>".format(num)


def form_sentence(prem: str, hypo: str,
                  pattern_idx: int, num_tokens_per_pattern: int,
                  only_sep: bool, space: str = ' '):
    next_contoken = pattern_idx * num_tokens_per_pattern
    contokens = [
        contoken(i)
        for i in range(next_contoken, next_contoken+num_tokens_per_pattern)
    ]

    if only_sep:
        before, after = '', ''
        center = space.join(contokens)
    else:
        num_outer = num_inner = num_tokens_per_pattern // 3
        num_inner += num_tokens_per_pattern % 3

        center = space.join(contokens[num_outer:num_outer+num_inner])
        before = space.join(contokens[:num_outer])
        after = space.join(contokens[num_outer+num_inner:])

    if before:
        before = before + space
        after = space + after

    sentence = before + prem + space + center + space + hypo + after
    return re.sub(r'\s+', space, sentence)


def choose_examples(examples_A, examples_B, is_reversed: bool):
    a, b = 0, 0
    if examples_A[a] == examples_B[b]:
        if len(examples_B) > 1:
            b = 1
        else:
            a = 1

    if is_reversed:
        return examples_B[b], examples_A[a]
    else:
        return examples_A[a], examples_B[b]


def negate(verb_phrase: str) -> str:
    tokens = re.split(r'\s+', verb_phrase)
    if tokens[0] in ['is', 'are', 'were', 'was']:
        new_tokens = tokens[:1] + ['not'] + tokens[1:]
    else:
        if tokens[0].endswith('s'):
            new_tokens = ['does', 'not', tokens[0][:-1]] + tokens[1:]
        else:
            new_tokens = ['do', 'not', tokens[0][:-1]] + tokens[1:]
    return ' '.join(new_tokens)


def mask_equivalent(self, string: str, mask_token, tokenizer, add_space=True) -> str:
    longer_string = mask_token
    if add_space:
        longer_string = longer_string + ' '
    longer_string = longer_string + string.strip()
    num_tokens = len(
        tokenizer.encode(longer_string, add_special_tokens=False)
    ) - 1
    return " ".join([mask_token] * num_tokens)


T = TypeVar('T')


def chunks(lst: List[T], n: int) -> Iterable[List[T]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunks_from_iterable(iterable: Iterable[T], size: int) -> Iterable[Sequence[T]]:
    """Generate adjacent chunks of data"""
    it = iter(iterable)
    return iter(lambda: tuple(itertools.islice(it, size)), ())
