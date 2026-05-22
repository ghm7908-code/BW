import torch
from torch import nn
import torch.nn.functional as F
from utils.geometry_utils import edge_acc

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class CornerCriterion(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.loss_rate = 9

    def forward(self, outputs_s1, targets, gauss_targets, epoch=0):
        outputs_s1 = torch.nan_to_num(outputs_s1, nan=0.5, posinf=1.0, neginf=0.0)
        outputs_s1 = outputs_s1.clamp(1e-6, 1.0 - 1e-6)
        gauss_targets = torch.nan_to_num(gauss_targets.float(), nan=0.0, posinf=1.0, neginf=0.0)
        gauss_targets = gauss_targets.clamp(0.0, 1.0)

        preds_s1 = (outputs_s1 >= 0.5).float()
        pos_target_ids = torch.where(targets == 1)
        if len(pos_target_ids[0]) > 0:
            correct = (preds_s1[pos_target_ids] == targets[pos_target_ids]).float().sum()
            recall_s1 = correct / len(pos_target_ids[0])
        else:
            recall_s1 = torch.zeros((), device=outputs_s1.device)

        rate = self.loss_rate

        loss_weight = (gauss_targets > 0.5).float() * rate + 1
        loss_s1 = F.binary_cross_entropy(outputs_s1, gauss_targets, weight=loss_weight, reduction='none')
        loss_s1 = loss_s1.sum(-1).sum(-1).mean()

        return loss_s1, recall_s1


class RoofStructureCriterion(nn.Module):
    def __init__(self, pos_weight=6.0, dice_weight=1.0, eps=1e-6):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, outputs, targets):
        if outputs.dim() == 4:
            outputs = outputs.squeeze(1)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)
        targets = torch.nan_to_num(targets.float(), nan=0.0, posinf=1.0, neginf=0.0)
        targets = targets.clamp(0.0, 1.0)

        weights = 1.0 + targets * self.pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, weight=weights, reduction='mean')

        probs = torch.sigmoid(outputs)
        outputs_flat = probs.flatten(1)
        targets_flat = targets.flatten(1)
        intersection = (outputs_flat * targets_flat).sum(dim=1)
        denom = outputs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (denom + self.eps)

        return bce_loss + self.dice_weight * dice_loss.mean()


class Corner3dCriterion(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.loss_rate = 9
        self.matcher = matcher
        self.focal_alpha = 0.25
        self.num_classes = 1

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    def loss_3dcenter(self, pred_centers, targets, indices, corner_logits):
        

        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = pred_centers[:, :, 2][idx]
        target_3dcenter = torch.cat([t['coords'][:, 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_len =  torch.cat([t['length'] for t in targets], dim=0)
        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')

        losses = loss_3dcenter.sum() / target_len.sum()

        return losses
    
    def loss_labels(self, pred_logits, targets, indices):        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.full(pred_logits.shape, self.num_classes-1,
                                    dtype=torch.float32, device=pred_logits.device)
        
        target_classes[idx] = target_classes_o
        
        loss_ce = F.binary_cross_entropy_with_logits(pred_logits, target_classes)
        pred_probs = torch.sigmoid(pred_logits)
        gamma=2 
        alpha=0.25
        focal_weights = alpha * target_classes * (1 - pred_probs) ** gamma + (1 - alpha) * (1 - target_classes) * pred_probs ** gamma
        
        loss_ce = torch.mean(focal_weights * loss_ce)

        return loss_ce
    @torch.no_grad()
    def loss_height(self, pred_centers, targets, indices):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = pred_centers[:, :, 2][idx]
        target_3dcenter = torch.cat([t['coords'][:, 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_len =  torch.cat([t['length'] for t in targets], dim=0)
        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')

        losses = loss_3dcenter.sum() / target_len.sum()

        return losses
    @torch.no_grad()
    def loss_x(self, pred_centers, targets, indices):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = pred_centers[:, :, 0][idx]
        target_3dcenter = torch.cat([t['coords'][:, 0][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_len =  torch.cat([t['length'] for t in targets], dim=0)
        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')

        losses = loss_3dcenter.sum() / target_len.sum()

        return losses
    @torch.no_grad()
    def loss_y(self, pred_centers, targets, indices):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = pred_centers[:, :, 1][idx]
        target_3dcenter = torch.cat([t['coords'][:, 1][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_len =  torch.cat([t['length'] for t in targets], dim=0)
        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')

        losses = loss_3dcenter.sum() / target_len.sum()

        return losses

    @torch.no_grad()
    def loss_cardinality(self, pred_logits, targets):
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([(v["length"]) for v in targets], device=device) 
        card_pred = (pred_logits.sigmoid() > 0.5).flatten(1, 2).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return card_err
    
    def forward(self, corner_logits, corner_coord, targets):
        indices = self.matcher(corner_coord, corner_logits, targets)
        loss_3dcenter  = self.loss_3dcenter(corner_coord, targets, indices,corner_logits )
        loss_labels = self.loss_labels(corner_logits, targets, indices)
        loss_cardinality = self.loss_cardinality(corner_logits, targets)
        loss_height = self.loss_height(corner_coord, targets, indices)
        loss_x = self.loss_x(corner_coord, targets, indices)
        loss_y = self.loss_y(corner_coord, targets, indices)
        return loss_3dcenter, loss_labels, loss_cardinality, loss_height, loss_x, loss_y


class EdgeCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('edge_loss_weight', torch.tensor([0.33, 1.0], dtype=torch.float32))

    def forward(self, logits_s1, logits_s2_hybrid, logits_s2_rel, s2_ids, s2_edge_mask, edge_labels, edge_lengths,
                edge_mask, s2_gt_values):
        s1_losses = F.cross_entropy(logits_s1, edge_labels, weight=self.edge_loss_weight, reduction='none')
        s1_losses[torch.where(edge_mask == True)] = 0
        s1_losses = s1_losses[torch.where(s1_losses > 0)].sum() / edge_mask.shape[0]
        gt_values = torch.ones_like(edge_mask).long() * 2
        s1_acc = edge_acc(logits_s1, edge_labels, edge_lengths, gt_values)

        s2_labels = torch.gather(edge_labels, 1, s2_ids)

        s2_losses_hybrid = F.cross_entropy(logits_s2_hybrid, s2_labels, weight=self.edge_loss_weight, reduction='none')
        s2_losses_hybrid[torch.where((s2_edge_mask == True) | (s2_gt_values != 2))] = 0
        s2_losses_hybrid = s2_losses_hybrid[torch.where(s2_losses_hybrid > 0)].sum() / s2_edge_mask.shape[0]
        s2_edge_lengths = (s2_edge_mask == 0).sum(dim=-1)
        s2_acc_hybrid = edge_acc(logits_s2_hybrid, s2_labels, s2_edge_lengths, s2_gt_values)

        s2_losses_rel = F.cross_entropy(logits_s2_rel, s2_labels, weight=self.edge_loss_weight, reduction='none')
        s2_losses_rel[torch.where((s2_edge_mask == True) | (s2_gt_values != 2))] = 0
        s2_losses_rel = s2_losses_rel[torch.where(s2_losses_rel > 0)].sum() / s2_edge_mask.shape[0]
        s2_edge_lengths = (s2_edge_mask == 0).sum(dim=-1)
        s2_acc_rel = edge_acc(logits_s2_rel, s2_labels, s2_edge_lengths, s2_gt_values)

        return s1_losses, s1_acc, s2_losses_hybrid, s2_acc_hybrid, s2_losses_rel, s2_acc_rel
