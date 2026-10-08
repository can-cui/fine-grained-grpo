"""Fairseq task registration for word-final pause prediction."""

from fairseq.tasks import FairseqTask, register_task
from fairseq.data.pause_lmdb_dataset import PauseLMDBDataset


@register_task("pause_prediction")
class PausePredictionTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="pause LMDB root containing train/valid/test")
        # 公司原定制 train.py 会无条件读取 args.bert_pretrain。V1 不使用
        # BERT，但仍需提供空默认值以兼容该训练入口。
        parser.add_argument("--bert-pretrain", default="")
        # 当前训练环境使用的 Fairseq Trainer 会直接读取这两个字段，但其
        # 全局 options.py 未注册它们。由 V4 task 提供兼容默认值，避免在
        # 创建 epoch iterator 时因 Namespace 缺字段而中断。
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
