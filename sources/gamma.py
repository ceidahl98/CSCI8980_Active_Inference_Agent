from torch import nn
import torch
import utils
from .source import Source
class Gamma(Source):
    def __init__(self, nvars, concentration,rate, scale=1):
        super().__init__(nvars)
        self.nvars = nvars
        self.nvars_higher = nvars.copy()
        self.nvars_higher[1] = nvars[1]//8
        self.nvars_higher[2] = nvars[2]//8
        self.register_buffer(
            'scale', torch.tensor(scale, dtype=torch.get_default_dtype()))

        self.register_buffer(
            'concentration',torch.tensor(concentration,dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'rate', torch.tensor(rate,dtype=torch.get_default_dtype())
        )
        self.gamma_dist = torch.distributions.Gamma(concentration=self.concentration,rate=self.rate)


    def sample(self,batch_size):
        shape = [batch_size]+ self.nvars

        shape_higher = [batch_size//4] + [4]+ self.nvars_higher
        shape_higher = [shape_higher[i] for i in [0, 2, 1, 3, 4]]
        z_lower = self.gamma_dist.rsample(shape)
        z_lower,_ = utils.exp_inverse(z_lower)
        z_higher = self.gamma_dist.rsample(shape_higher)
        return z_lower, z_higher

    def log_prob(self, z):
        z = torch.clamp(z,min=1e-12)
        log_prob = self.gamma_dist.log_prob(z).to(z.device)

        return log_prob.view(log_prob.shape[0],-1).sum(dim=-1)