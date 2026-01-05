import torch, math
import numpy as np
import torch.nn as nn

def validate_model(model, dataset, device, n_val_tasks=100):
    criterion = nn.MSELoss(reduction='mean')
    model.eval()
    losses, corrs = [], []
    with torch.no_grad():
        for _ in range(n_val_tasks):
            t = dataset.create_meta_task()

            ctx_x = t['ctx_x'].unsqueeze(0).float().to(device)
            ctx_z = t['ctx_z'].unsqueeze(0).float().to(device)
            qry_x = t['qry_x'].unsqueeze(0).float().to(device)
            qry_z = t['qry_z'].unsqueeze(0).float().to(device)
            t_ctx = torch.tensor(t['endpoints']['ctx'], dtype=torch.long).unsqueeze(0).to(device)  # [1,C]
            t_qry = torch.tensor(t['endpoints']['qry'], dtype=torch.long).unsqueeze(0).to(device)  # [1,Q]
            
            pred = model(ctx_x, ctx_z, qry_x, t_ctx, t_qry)
            # pred, log_sigma2 = model(ctx_x, ctx_z, qry_x, t_ctx, t_qry)
            # loss = gaussian_nll_loss(mu, log_sigma2, qry_z)
            
            loss = criterion(pred, qry_z)
            losses.append(loss.item())

            p = pred.flatten().float().cpu()
            q = qry_z.flatten().float().cpu()
            if p.numel() > 1 and torch.var(p) > 0 and torch.var(q) > 0:
                r = torch.corrcoef(torch.stack([p, q]))[0, 1]
                corr = r.item() if torch.isfinite(r) else 0.0
            else:
                corr = 0.0
            corrs.append(corr)

    model.train()
    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'loss_std': float(np.std(losses)) if losses else 0.0,
        'correlation': float(np.mean(corrs)) if corrs else 0.0,
        'correlation_std': float(np.std(corrs)) if corrs else 0.0,
    }

def get_opt_lr_schedule(model, config):
    warmup_tasks, total_tasks = config['warmup_tasks'], config['total_tasks']

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=1e-4, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8) 

    def lr_lambda(step):
        if step < warmup_tasks:
            return step / warmup_tasks
        else:
            progress = (step - warmup_tasks) / (total_tasks - warmup_tasks)
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))  # Decay to 10%
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def gaussian_nll_loss(mu, log_sigma2, targets):
    """ Average loss over all query patches and horizons """
    sigma2 = torch.exp(log_sigma2)
    log_2pi = math.log(2 * math.pi)
    
    loss_per_point = 0.5 * (log_2pi + log_sigma2 + ((targets - mu) ** 2) / sigma2)
    
    return loss_per_point.mean()