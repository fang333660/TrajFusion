# utils/EMA.py (完整工业版)
import torch

class EMAHelper:
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, module):
        """
        注册需要进行EMA的模型。
        这会创建模型参数的一个“影子”副本。
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """
        在每个训练步骤（optimizer.step()之后）调用，
        用于更新影子的权重。
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def ema(self, module):
        """
        将模型的当前权重替换为EMA的平均权重。
        在进行评估或保存EMA模型时调用。
        """
        # 首先，备份原始权重
        self.original = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, module):
        """
        将模型的权重从EMA权重恢复为原始权重。
        在评估或保存完成后调用，以便继续训练。
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                if name in self.original:
                    param.data.copy_(self.original[name])
                else:
                    # This case should not happen if ema() was called before.
                    # It's a safeguard.
                    pass 
        self.original = {} # 清空备份