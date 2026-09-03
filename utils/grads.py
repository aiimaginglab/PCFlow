import torch
import torch.nn.functional as F

def collect_grad_dict(model): 
    """
    create gradient dictionary
    """
    grad_dict = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_dict[name] = p.grad.detach().clone()
    return grad_dict

def accumulate_grad(accum, new):
    if accum is None:
        return {k: v.clone() for k, v in new.items()}
    for k in new:
        accum[k] += new[k]
    return accum

def flatten_grad_dict(grad_dict):
    return torch.cat([
        grad_dict[k].view(-1)
        for k in sorted(grad_dict.keys())
    ])

def virtual_update(theta, grad_dict, eta):
    return {
        k: theta[k] - eta * grad_dict[k]
        for k in grad_dict
    }

def delta_theta(theta_new, theta_old):
    return torch.cat([
        (theta_new[k] - theta_old[k]).view(-1)
        for k in theta_new
    ])

def t_to_logsnr(t, sigma_min=1e-5):
    alpha2 = t ** 2
    sigma2 = (1 - (1 - sigma_min) * t) ** 2
    return torch.log(alpha2 / sigma2 + 1e-12)

def ratio(gradient_a_flatten, gradient_b_flatten):
    ratio_value = gradient_a_flatten.norm() / (gradient_b_flatten.norm() + 1e-8)
    return ratio_value

def cosine_sim(g1, g2):
    return F.cosine_similarity(g1, g2, dim=0)