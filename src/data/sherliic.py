from typing import Sequence, List, Dict, Union, Tuple
from torch.utils.data import Dataset
import csv
from tqdm import tqdm
import logging
from .common import LABEL_KEY, SENT_KEY, ANTI_KEY,\
    choose_examples, chunks, form_sentence


def unpack_row(row: Sequence[str])\
    -> Tuple[
        int, int, int, int, int,
        str, str, str, str,
        str, str, str, str,
        bool, bool,
        Tuple[str, str, str], Tuple[str, str, str],
        bool, float, float,
        float, int]:
    sample_id, prem_type, prem_rel, hypo_type, hypo_rel,\
        prem_argleft, prem_middle, prem_argright, prem_end,\
        hypo_argleft, hypo_middle, hypo_argright, hypo_end,\
        is_prem_reversed, is_hypo_reversed,\
        examples_A, examples_B, gold_label,\
        rel_score, sign_score, esr_score,\
        num_disagr = row

    return int(sample_id), int(prem_type), int(prem_rel), int(hypo_type), int(hypo_rel),\
        prem_argleft, prem_middle, prem_argright, prem_end,\
        hypo_argleft, hypo_middle, hypo_argright, hypo_end,\
        is_prem_reversed == 'True', is_hypo_reversed == 'True',\
        tuple(examples_A.split(' / ')), tuple(examples_B.split(' / ')),\
        gold_label == 'yes', float(rel_score), float(sign_score),\
        float(esr_score), int(num_disagr)


class Sherliic(Dataset):
    def __init__(
            self, path_to_csv: str, num_patterns: int = 1,
            num_tokens_per_pattern: int = 1, only_sep=True,
            use_antipatterns=False,
            training: bool = False, pattern_chunk_size: int = 5
    ):
        self.training = training
        self.pattern_chunk_size = pattern_chunk_size
        self.only_sep = only_sep
        self.num_patterns = num_patterns
        self.num_tokens_per_pattern = num_tokens_per_pattern
        self.use_antipatterns = use_antipatterns

        self.data = self.load_dataset(path_to_csv)

    def load_dataset(self, fname):
        logger = logging.getLogger(__name__)
        logger.info('Loading dataset from {}'.format(fname))
        with open(fname) as f:
            cr = csv.reader(f)
            next(cr)  # headers
            data = [inst for row in tqdm(cr)
                    for inst in self.create_instances(*unpack_row(row))]
        return data

    def create_sentence(self, pattern_idx: int, premise, hypothesis,
                        is_prem_reversed, is_hypo_reversed,
                        examples_A, examples_B) -> str:
        prem_argleft, prem_middle, prem_argright, prem_end = premise
        hypo_argleft, hypo_middle, hypo_argright, hypo_end = hypothesis

        prem_argleft, prem_argright = choose_examples(
            examples_A, examples_B, is_prem_reversed)
        hypo_argleft, hypo_argright = choose_examples(
            examples_A, examples_B, is_hypo_reversed)

        prem_phrase = "{} {} {}".format(
            prem_argleft, prem_middle, prem_argright+prem_end)
        hypo_phrase = "{} {} {}".format(
            hypo_argleft, hypo_middle, hypo_argright)

        sentence = form_sentence(prem_phrase, hypo_phrase, pattern_idx,
                                 self.num_tokens_per_pattern, self.only_sep)

        return sentence

    def create_single_instance(
            self, premise: Tuple[str, str, str, str],
            hypothesis: Tuple[str, str, str, str],
            is_prem_reversed: bool, is_hypo_reversed: bool,
            examples_A: Tuple[str, str, str],
            examples_B: Tuple[str, str, str], gold_label: bool
    ) -> Dict[str, Union[bool, str, List[str]]]:
        inst = {
            LABEL_KEY: 1 if gold_label else 0
        }

        inst[SENT_KEY] = []
        for pattern_idx in range(self.num_patterns):
            inst[SENT_KEY].append(
                self.create_sentence(
                    pattern_idx,
                    premise, hypothesis,
                    is_prem_reversed, is_hypo_reversed,
                    examples_A, examples_B
                )
            )

        if self.use_antipatterns:
            inst[ANTI_KEY] = []
            for pattern_idx in range(self.num_patterns, 2*self.num_patterns):
                inst[ANTI_KEY].append(
                    self.create_sentence(
                        pattern_idx,
                        premise, hypothesis,
                        is_prem_reversed, is_hypo_reversed,
                        examples_A, examples_B
                    )
                )

        return inst

    def create_instances(
            self, sample_id: int, prem_type: int, prem_rel: int, hypo_type: int, hypo_rel: int,
            prem_argleft: str, prem_middle: str, prem_argright: str, prem_end: str,
            hypo_argleft: str, hypo_middle: str, hypo_argright: str, hypo_end: str,
            is_prem_reversed: bool, is_hypo_reversed: bool,
            examples_A: Tuple[str, str, str], examples_B: Tuple[str, str, str], gold_label: bool,
            rel_score: float, sign_score: float, esr_score: float, num_disagr: int
    ) -> List[Dict[str, Union[bool, str, List[str]]]]:

        instances = []

        inst = self.create_single_instance(
            (prem_argleft, prem_middle, prem_argright, prem_end),
            (hypo_argleft, hypo_middle, hypo_argright, hypo_end),
            is_prem_reversed, is_hypo_reversed,
            examples_A, examples_B,
            gold_label
        )
        if self.training:
            lists = self.unpack_instance(inst)
            label = lists[-1]
            chunked = [chunks(x, self.pattern_chunk_size)
                       for x in lists[:-1]]
            for chunk in zip(*chunked):
                smaller_inst = {
                    SENT_KEY: chunk[0],
                    LABEL_KEY: label
                }
                if self.use_antipatterns:
                    smaller_inst[ANTI_KEY] = chunk[1]
                instances.append(smaller_inst)
        else:
            instances.append(inst)

        return instances

    def unpack_instance(
            self, inst: Dict[str, Union[str, bool, List[str]]]
    ) -> List[Union[str, bool, List[str]]]:
        if self.use_antipatterns:
            return [inst[SENT_KEY], inst[ANTI_KEY], inst[LABEL_KEY]]
        else:
            return [inst[SENT_KEY], inst[LABEL_KEY]]

    def __getitem__(self, index):
        inst = self.data[index]
        if self.use_antipatterns:
            anti = inst[ANTI_KEY]
        else:
            anti = None

        return inst[SENT_KEY], anti, inst[LABEL_KEY]

    def __len__(self):
        return len(self.data)
