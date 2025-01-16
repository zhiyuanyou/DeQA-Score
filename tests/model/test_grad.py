import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, dim_in, closeset):
        super().__init__()
        self.closeset = closeset
        self.lm_head = nn.Linear(dim_in, 1024)
        self.level_ids = [128, 256, 512, 640, 768]

    def forward(self, x_A, x_B, gt):
        logits_A = self.lm_head(x_A)  # [B, V]
        scores_A, loss_inlevel_A = self.get_score(logits_A)  # [B, ]
        logits_B = self.lm_head(x_B)  # [B, V]
        scores_B, loss_inlevel_B = self.get_score(logits_B)  # [B, ]
        loss = self.rating_loss(scores_A, scores_B, gt)
        return loss

    def get_score(self, logits):
        probs_org = torch.softmax(logits, dim=1)  # [B, V]
        loss_in_level = 1 - probs_org[:, self.level_ids].contiguous().sum(dim=1)  # [B, 5] -> [B, ]
        loss_in_level = loss_in_level.mean()  # level prob > 0.99

        if self.closeset:
            logits_levels = logits[:, self.level_ids].contiguous()
            probs = torch.softmax(logits_levels, dim=1)
        else:
            probs = probs_org[:, self.level_ids].contiguous()
        weights = torch.tensor([5., 4., 3., 2., 1.]).to(probs)
        scores = torch.matmul(probs, weights)
        return scores, loss_in_level

    def rating_loss(self, pred_scores_A, pred_scores_B, gt):
        pred = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / 2))  # 2 -> sqrt(2 * (1**2 + 1**2))
        # eps=1e-8 is important. eps=0 is unable to step, and lr keeps unchanged. 
        eps = 1e-8
        loss = (1 - (pred * gt + eps).sqrt() - ((1 - pred) * (1 - gt) + eps).sqrt()).mean()
        return loss


if __name__ == "__main__":
    dim_in = 512
    closeset = False
    batch_size = 8
    num_epoch = 5
    num_step = 100
    lr = 0.1

    model = MyModel(dim_in, closeset)
    parameters = model.parameters()
    optim = torch.optim.AdamW(
        parameters,
        lr = lr,
    )

    for epoch in range(num_epoch):
        for step in range(num_step):
            x_A = torch.rand(batch_size, dim_in)
            x_B = torch.rand(batch_size, dim_in)
            gt = torch.rand(batch_size)
            loss = model(x_A, x_B, gt)

            optim.zero_grad()
            loss.backward()
            optim.step()
            print("=" * 100)
            print(model.lm_head.weight.grad)
            print(model.lm_head.weight)
