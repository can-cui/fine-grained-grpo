"""为停顿分类头和部分解冻 wav2vec 保持独立学习率的 Fairseq Adam。"""

import ast
import json
from pathlib import Path

import torch
from fairseq.optim import FairseqOptimizer, register_optimizer


@register_optimizer("pause_adam")
class PauseAdam(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        parameters = [parameter for parameter in params if parameter.requires_grad]
        untagged = [parameter for parameter in parameters if not hasattr(parameter, "_pause_lr_group")]
        if untagged:
            raise ValueError(f"pause_adam 发现 {len(untagged)} 个未标注的可训练参数")
        head = [p for p in parameters if p._pause_lr_group == "head"]
        wav2vec = [p for p in parameters if p._pause_lr_group == "wav2vec"]
        if not head or not wav2vec:
            raise ValueError(
                f"pause_adam 参数组不完整：head={len(head)}, wav2vec={len(wav2vec)}"
            )
        self.head_base_lr = float(args.lr[0])
        self.wav2vec_base_lr = float(args.wav2vec_lr)
        if self.head_base_lr <= 0 or self.wav2vec_base_lr <= 0:
            raise ValueError("分类头与 wav2vec 学习率必须为正数")
        self.wav2vec_lr_ratio = self.wav2vec_base_lr / self.head_base_lr
        self.audit_path = Path(args.save_dir) / "optimizer_update_audit.json"
        self._first_step_audited = False
        betas = ast.literal_eval(args.adam_betas)
        groups = [
            {"params": head, "lr": self.head_base_lr, "pause_group": "head"},
            {"params": wav2vec, "lr": self.wav2vec_base_lr, "pause_group": "wav2vec"},
        ]
        self._optimizer = torch.optim.Adam(
            groups,
            betas=betas,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        print(json.dumps({
            "optimizer": "pause_adam",
            "head_parameter_tensors": len(head),
            "wav2vec_parameter_tensors": len(wav2vec),
            "head_lr": self.head_base_lr,
            "wav2vec_lr": self.wav2vec_base_lr,
            "wav2vec_lr_ratio": self.wav2vec_lr_ratio,
        }, ensure_ascii=False))

    @staticmethod
    def add_args(parser):
        parser.add_argument("--adam-betas", default="(0.9, 0.999)")
        parser.add_argument("--adam-eps", type=float, default=1e-8)
        parser.add_argument("--weight-decay", "--wd", type=float, default=0.0)
        parser.add_argument("--wav2vec-lr", type=float, default=1e-5)

    @property
    def optimizer_config(self):
        return {
            "lr": self.head_base_lr,
            "betas": ast.literal_eval(self.args.adam_betas),
            "eps": self.args.adam_eps,
            "weight_decay": self.args.weight_decay,
        }

    def set_lr(self, lr):
        head_lr = float(lr)
        for group in self._optimizer.param_groups:
            if group.get("pause_group") == "wav2vec":
                group["lr"] = head_lr * self.wav2vec_lr_ratio
            else:
                group["lr"] = head_lr

    def get_lr(self):
        for group in self._optimizer.param_groups:
            if group.get("pause_group") == "head":
                return group["lr"]
        raise RuntimeError("pause_adam 找不到 head 参数组")

    def step(self, closure=None):
        if self._first_step_audited:
            return self._optimizer.step(closure)
        snapshots = {
            group["pause_group"]: [parameter.detach().clone() for parameter in group["params"]]
            for group in self._optimizer.param_groups
        }
        result = self._optimizer.step(closure)
        audit = {}
        for group in self._optimizer.param_groups:
            group_name = group["pause_group"]
            before = snapshots[group_name]
            changed = sum(
                not torch.equal(old, parameter.detach())
                for old, parameter in zip(before, group["params"])
            )
            audit[group_name] = {
                "parameter_tensors": len(group["params"]),
                "changed_parameter_tensors": changed,
                "lr": float(group["lr"]),
            }
        audit["ok"] = all(values["changed_parameter_tensors"] > 0 for values in audit.values())
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not audit["ok"]:
            raise RuntimeError(f"首次 optimizer step 参数更新范围校验失败：{audit}")
        print(json.dumps({"optimizer_update_audit": str(self.audit_path), **audit}, ensure_ascii=False))
        self._first_step_audited = True
        return result
