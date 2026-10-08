"""Masked binary pause loss and pooled confusion-matrix metrics."""

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("pause_bce_loss")
class PauseBCECriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.positive_weight = float(args.positive_weight)
        self.pause_threshold = float(args.pause_threshold)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--positive-weight", type=float, default=1.0)
        parser.add_argument("--pause-threshold", type=float, default=0.5)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        logits = net_output["pause_logits"]
        targets = sample["target"].to(logits.dtype)
        mask = sample["valid_mask"].bool()
        sample_size = int(mask.long().sum().item())
        if sample_size == 0:
            raise ValueError("batch contains no valid pause targets")

        pos_weight = logits.new_tensor(self.positive_weight)
        element_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight
        )
        loss = element_loss.masked_select(mask).sum()

        predictions = torch.sigmoid(logits) >= self.pause_threshold
        positives = targets >= 0.5
        tp = ((predictions & positives) & mask).long().sum()
        fp = ((predictions & ~positives) & mask).long().sum()
        fn = ((~predictions & positives) & mask).long().sum()
        tn = ((~predictions & ~positives) & mask).long().sum()
        logging_output = {
            "loss": utils.item(loss.detach()),
            "tp": utils.item(tp),
            "fp": utils.item(fp),
            "fn": utils.item(fn),
            "tn": utils.item(tn),
            "ntokens": sample_size,
            "nsentences": int(sample.get("nsentences", targets.size(0))),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        tp = sum(log.get("tp", 0) for log in logging_outputs)
        fp = sum(log.get("fp", 0) for log in logging_outputs)
        fn = sum(log.get("fn", 0) for log in logging_outputs)
        tn = sum(log.get("tn", 0) for log in logging_outputs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        total = tp + fp + fn + tn
        return {
            "loss": sum(log.get("loss", 0.0) for log in logging_outputs) / max(sample_size, 1),
            "nll_loss": sum(log.get("loss", 0.0) for log in logging_outputs) / max(sample_size, 1),
            "pause_precision": precision,
            "pause_recall": recall,
            "pause_f1": f1,
            "pause_accuracy": (tp + tn) / total if total else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "ntokens": sample_size,
            "nsentences": sum(log.get("nsentences", 0) for log in logging_outputs),
            "sample_size": sample_size,
        }
