from torch import nn
import utils

class Flow(nn.Module):
    def __init__(self, prior=None):
        super().__init__()
        self.prior = prior

    def forward(self, x):
        raise NotImplementedError(str(type(self)))

    def inverse(self, z,z_lower):
        raise NotImplementedError(str(type(self)))

    def sample(self, batch_size,device, prior=None):
        if prior is None:
            prior = self.prior
        assert prior is not None
        print(batch_size,"BATCH_SIZE")
        z_lower,z_higher = prior.sample(batch_size)
        logp = prior.log_prob(z_higher).to(device)
        z_lower,_ = utils.exp_inverse(z_lower)
        z_higher,_ = utils.exp_inverse(z_higher)
        x, logp_ = self.inverse(z_higher.to(device),z_lower.to(device))
        return x, logp - logp_.to(device)

    def log_prob(self, x):

        z_lower,z_higher, logp,upper_logp = self.forward(x) #z_higher is a list, z_lower is a single tensor




        z_higher.append(z_lower)
        all_z = z_higher.copy()
        mean = sum([z.mean().item() for z in all_z])/len(all_z)
        maximum = max([z.max().item() for z in all_z])
        minimum = max([z.min().item() for z in all_z])
        print(mean,"MEAN")
        print(maximum,"MAX")
        print(minimum,"MIN")


        #print(logp.mean()/(1*32*32),"LDJ")
        probs_higher = []
        if self.prior is not None:

            prob_lower = self.prior.log_prob(z_lower)
            for prob,upper_z in zip(upper_logp,z_higher):
                prob_higher = self.prior.log_prob(upper_z)
                probs_higher.append(prob_higher.to(x.device))
            #print(prob.mean()/(1*32*32), "LOG PROB")


        return prob_lower,logp,probs_higher,upper_logp
