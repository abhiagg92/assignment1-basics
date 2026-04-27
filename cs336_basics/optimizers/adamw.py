from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                
                grad = p.grad.data
                lr_t = lr*(math.sqrt(1-beta2**t)/(1-beta1**t)) 
                p.data -= lr*weight_decay*p.data
                m  = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*torch.square(grad)
                p.data -= lr_t*m/(torch.sqrt(v)+eps)
                
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
