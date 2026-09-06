import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import numpy as np
import torch.nn.functional as F


class NCEAverage_each_pre(nn.Module):
    def __init__(self, inputSize, gpu_ids, g_data, m_data,
                 memory_feat_gbm, memory_prob_gbm, memory_name_gbm,
                 memory_feat_mate, memory_prob_mate, memory_name_mate,
                 alpha, K, T=0.07, momentum=0.5, use_softmax=False):

        super(NCEAverage_each_pre, self).__init__()

        self.gpu_ids = gpu_ids
        self.g_data = g_data
        self.m_data = m_data


        self.alpha = alpha
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))

        self.register_buffer('memory_feat_gbm', memory_feat_gbm)
        self.register_buffer('memory_prob_gbm', memory_prob_gbm)
        self.memory_name_gbm = memory_name_gbm

        self.register_buffer('memory_feat_mate', memory_feat_mate)
        self.register_buffer('memory_prob_mate', memory_prob_mate)
        self.memory_name_mate = memory_name_mate

        self.memory_feat_gbm = nn.functional.normalize(self.memory_feat_gbm, dim=1)
        self.memory_feat_mate = nn.functional.normalize(self.memory_feat_mate, dim=1)

    def sample_positive_index(self, memory_prob, memory_name, cur_name):
        memory_prob = memory_prob.view(-1)
        device = memory_prob.device

        candidate_idx = torch.nonzero(memory_prob > self.alpha, as_tuple=True)[0]

        if cur_name in memory_name:
            self_idx = memory_name.index(cur_name)
            candidate_idx = candidate_idx[candidate_idx != self_idx]
        else:
            self_idx = None

        # If no same-class sample satisfies p_true > alpha,
        # sample from the top quarter of the sorted class-specific bank.
        if candidate_idx.numel() == 0:
            top_k = max(1, len(memory_name) // 4)
            candidate_idx = torch.arange(top_k, device=device)

            if self_idx is not None:
                candidate_idx = candidate_idx[candidate_idx != self_idx]

        # If the top quarter only contains itself, fall back to all same-class non-self samples.
        if candidate_idx.numel() == 0:
            candidate_idx = torch.arange(len(memory_name), device=device)

            if self_idx is not None:
                candidate_idx = candidate_idx[candidate_idx != self_idx]

        rand = torch.randint(candidate_idx.numel(), (1,), device=device)
        return candidate_idx[rand]

    @torch.no_grad()
    def update_reference_banks(self, feat, label, out, name):
        """Refresh the reference banks after a complete training epoch."""
        feat = F.normalize(feat.detach(), dim=1)
        label = label.detach().long()
        out = out.detach()
        momentum = self.params[4].item()

        for sample_idx, cur_name in enumerate(name):
            class_idx = int(label[sample_idx].item())
            confidence = out[sample_idx, class_idx]

            if class_idx == 0:
                memory_feat = self.memory_feat_gbm
                memory_prob = self.memory_prob_gbm
                memory_name = self.memory_name_gbm
            else:
                memory_feat = self.memory_feat_mate
                memory_prob = self.memory_prob_mate
                memory_name = self.memory_name_mate

            bank_idx = memory_name.index(cur_name)
            if confidence < self.alpha:
                updated_feat = (
                    momentum * feat[sample_idx]
                    + (1.0 - momentum) * memory_feat[bank_idx]
                )
                updated_feat = F.normalize(updated_feat.unsqueeze(0), dim=1).squeeze(0)
            else:
                updated_feat = feat[sample_idx]

            memory_feat[bank_idx].copy_(updated_feat)
            memory_prob[bank_idx].copy_(confidence)

        self._sort_reference_bank(self.memory_feat_gbm, self.memory_prob_gbm,
                                  self.memory_name_gbm)
        self._sort_reference_bank(self.memory_feat_mate, self.memory_prob_mate,
                                  self.memory_name_mate)

    @staticmethod
    def _sort_reference_bank(memory_feat, memory_prob, memory_name):
        order = torch.argsort(memory_prob, descending=True)
        memory_feat.copy_(memory_feat.index_select(0, order))
        memory_prob.copy_(memory_prob.index_select(0, order))
        sorted_names = [memory_name[idx] for idx in order.cpu().tolist()]
        memory_name[:] = sorted_names

    @staticmethod
    def sample_reference_indices(memory_feat, num_samples):
        """Sample a fixed number of individual references from one class bank."""
        bank_size = memory_feat.size(0)
        if bank_size == 0:
            raise ValueError("Cannot sample RPCM references from an empty reference bank.")
        if num_samples <= bank_size:
            return torch.randperm(bank_size, device=memory_feat.device)[:num_samples]
        return torch.randint(bank_size, (num_samples,), device=memory_feat.device)

    def forward(self, feat, label, out, name, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()

        feat = nn.functional.normalize(feat, dim=1)

        batchSize = feat.size(0)
        outputSize = self.memory_feat_gbm.size(0)
        inputSize = self.memory_feat_gbm.size(1)

        # Negative sampling remains random.
        if idx is None:
            idx_n_g = self.multinomial_n_g.draw(self.K * 1).view(self.K, -1)
            idx_n_m = self.multinomial_n_m.draw(self.K * 1).view(self.K, -1)

        out_feature_p_n = torch.zeros(batchSize, K + 1).cuda(self.gpu_ids)

        # GBM samples: positive from GBM bank, negative from SBM bank.
        index_gbm = torch.nonzero((label == 0).int(), as_tuple=True)[0]

        for ind in index_gbm:
            ind_int = int(ind.item())
            cur_name = name[ind_int]

            feat_c = feat[ind].view(1, inputSize)

            idx_p_g = self.sample_positive_index(self.memory_prob_gbm, self.memory_name_gbm, cur_name )

            weight_feature_p = torch.index_select( self.memory_feat_gbm, 0,idx_p_g.view(-1) ).detach()

            weight_feature_n = torch.index_select( self.memory_feat_mate, 0,idx_n_m.view(-1) ).detach()

            l_pos = F.cosine_similarity(feat_c, weight_feature_p, dim=1)
            l_neg = F.cosine_similarity(feat_c, weight_feature_n, dim=1)

            out_feature_p_n[ind, :] = torch.cat([l_pos, l_neg], dim=0)

        # SBM samples: positive from SBM bank, negative from GBM bank.
        index_mate = torch.nonzero((label == 1).int(), as_tuple=True)[0]

        for ind in index_mate:
            ind_int = int(ind.item())
            cur_name = name[ind_int]

            feat_c = feat[ind].view(1, inputSize)

            idx_p_m = self.sample_positive_index(self.memory_prob_mate,  self.memory_name_mate, cur_name)

            weight_feature_p = torch.index_select(self.memory_feat_mate, 0,idx_p_m.view(-1) ).detach()

            weight_feature_n = torch.index_select(self.memory_feat_gbm, 0, idx_n_g.view(-1) ).detach()

            l_pos = F.cosine_similarity(feat_c, weight_feature_p, dim=1)
            l_neg = F.cosine_similarity(feat_c, weight_feature_n, dim=1)

            out_feature_p_n[ind, :] = torch.cat([l_pos, l_neg], dim=0)

        if self.use_softmax:
            out_feature_p_n = torch.div(out_feature_p_n, T)
            out_feature_p_n = out_feature_p_n.contiguous()

        else:
            out_feature_p_n = torch.exp(torch.div(out_feature_p_n, T))

            if Z_l < 0:
                self.params[2] = out_feature_p_n.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))

            out_feature_p_n = torch.div(out_feature_p_n, Z_l).contiguous()

        # RPCM jointly groups the current mini-batch with a fixed-size random
        # sample of individual reference features from each class-specific bank.
        index_clu_gbm = self.sample_reference_indices(self.memory_feat_gbm, batchSize)
        index_clu_mate = self.sample_reference_indices(self.memory_feat_mate, batchSize)

        feature_clu_gbm = torch.index_select(self.memory_feat_gbm, 0, index_clu_gbm).detach()
        feature_clu_mate = torch.index_select(self.memory_feat_mate, 0, index_clu_mate).detach()

        feature_clu_all = torch.cat([feature_clu_gbm, feature_clu_mate, feat], dim=0)

        gt_clu_all = torch.cat([ torch.zeros_like(index_clu_gbm), torch.ones_like(index_clu_mate),label ], dim=0)

        return out_feature_p_n, feature_clu_all, gt_clu_all

# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda(self.gpu_ids)
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda(self.gpu_ids)
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out

# if __name__ == '__main__':
#     NCE=NCEAverage_PRE(128, 400,10,0.07, 0.5, True)
#     X=torch.randn(4,128)
#     Y=torch.randn(4,128)
#     a=NCE(X,Y,None)
