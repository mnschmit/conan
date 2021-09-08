from typing import Tuple, Union, Dict, List, Iterable, Optional
from torch.utils.data import Dataset
from .common import LABEL_KEY, SENT_KEY, ANTI_KEY, form_sentence, chunks_from_iterable


class LevyHolt(Dataset):

    def __init__(
            self, txt_file: str, num_patterns: int = 1,
            num_tokens_per_pattern: int = 1, only_sep: bool = True,
            use_antipatterns: bool = False,
            training: bool = False, pattern_chunk_size: int = 5
    ):
        self.training = training
        self.pattern_chunk_size = pattern_chunk_size
        self.num_patterns = num_patterns
        self.num_tokens_per_pattern = num_tokens_per_pattern
        self.only_sep = only_sep
        self.use_antipatterns = use_antipatterns

        self.data = self.load_dataset(txt_file)

    def load_dataset(self, txt_file):
        data = []
        with open(txt_file) as f:
            for line in f:
                hypo, prem, label = line.strip().split('\t')
                hypo = tuple(h.strip() for h in hypo.split(','))
                prem = tuple(p.strip() for p in prem.split(','))
                label = label == 'True'
                data.extend(self.create_instances(prem, hypo, label))
        return data

    def create_sentence(self, pattern_idx: int, prem: Tuple[str, str, str],
                        hypo: Tuple[str, str, str]) -> str:
        sentence = form_sentence(
            ' '.join(prem), ' '.join(hypo),
            pattern_idx, self.num_tokens_per_pattern, self.only_sep
        )
        return sentence

    def create_single_instance(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool,
            pattern_indices: Iterable[int],
            antipattern_indices: Optional[Iterable[int]]
    ) -> Dict[str, Union[bool, str]]:
        inst = {}

        inst[SENT_KEY] = [
            self.create_sentence(pattern_idx, prem, hypo)
            for pattern_idx in pattern_indices
        ]
        if self.use_antipatterns:
            assert antipattern_indices is not None, "Internal Error"
            inst[ANTI_KEY] = [
                self.create_sentence(pattern_idx, prem, hypo)
                for pattern_idx in antipattern_indices
            ]
        inst[LABEL_KEY] = 1 if label else 0

        return inst

    def create_instances(
            self,
            prem: Tuple[str, str, str],
            hypo: Tuple[str, str, str],
            label: bool
    ) -> List[Dict[str, Union[bool, str]]]:
        instances = []

        if self.training:
            chunked = [
                chunks_from_iterable(
                    range(self.num_patterns), self.pattern_chunk_size),
                chunks_from_iterable(range(self.num_patterns, 2*self.num_patterns),
                                     self.pattern_chunk_size)
            ]
            for pattern_chunk, antipattern_chunk in zip(*chunked):
                inst = self.create_single_instance(
                    prem, hypo, label, pattern_chunk, antipattern_chunk)
                instances.append(inst)
        else:
            inst = self.create_single_instance(
                prem, hypo, label,
                range(self.num_patterns), range(self.num_patterns, 2*self.num_patterns))
            instances.append(inst)

        return instances

    def __getitem__(self, index):
        inst = self.data[index]
        if self.use_antipatterns:
            anti = inst[ANTI_KEY]
        else:
            anti = None

        return inst[SENT_KEY], anti, inst[LABEL_KEY]

    def __len__(self):
        return len(self.data)
