"""Fairseq task registration for word-final pause prediction."""

from fairseq.tasks import FairseqTask, register_task
from fairseq.data.pause_lmdb_dataset import PauseLMDBDataset


@register_task("pause_prediction")
class PausePredictionTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="pause LMDB root containing train/valid/test")
        # The company's customized train.py unconditionally reads args.bert_pretrain. V1 does not use
        # BERT, but must still provide an empty default value to remain compatible with this training entry point.
        parser.add_argument("--bert-pretrain", default="")
        # The Fairseq Trainer used in the current training environment reads these two fields directly, but its
        # global options.py does not register them. Provide compatible defaults in the V4 task to avoid
        # failures caused by missing Namespace fields when creating the epoch iterator.
        parser.add_argument("--buffer-size", type=int, default=0)
        parser.add_argument(
            "--grouped-shuffling", action="store_true", default=False
        )
        parser.add_argument(
            "--random-shuffle-data", action="store_true", default=False
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        self.datasets[split] = PauseLMDBDataset(
            self.args.data, split, shuffle=(split == "train")
        )

    def max_positions(self):
        return (10 ** 9, 10 ** 9)

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None
