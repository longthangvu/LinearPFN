import torch
import numpy as np
from torch.distributions import Beta, Uniform

from synthetic_prior.misc import dotdict

def sample_from_hyperpriors(hpp, series_len=12000, num_samples=1, device="cpu"):
    result = dotdict()

    for param, min_val, max_val, fixed_variance in [
        ("annual_param", hpp.a_min, hpp.a_max, hpp.a_fixed_variance),
        ("monthly_param", hpp.m_min, hpp.m_max, hpp.m_fixed_variance),
        ("weekly_param", hpp.w_min, hpp.w_max, hpp.w_fixed_variance),
        ("trend_lin", hpp.trend_lin_min, hpp.trend_lin_max, hpp.trend_lin_fixed_variance),
    ]:
        result[param] = triple_sampling(min_val=min_val,max_val=max_val,fixed_variance=fixed_variance,
                                        num_samples=num_samples,device=device)
    result.harmonics = torch.randint(hpp.harmonics_min, hpp.harmonics_max, (3,)).to(device)

    # make it equally likely to have a positive or negative exp trend
    mm = hpp.trend_exp_multiplier  
    f_exp = lambda x: 2 ** ((x - 1) * mm)
    f_exp_inv = lambda x: (torch.log2(x) / mm) + 1
    
    result.trend_exp = f_exp_inv(
        triple_sampling(min_val=f_exp(torch.scalar_tensor(hpp.trend_exp_min)),
                        max_val=f_exp(torch.scalar_tensor(hpp.trend_exp_max)),
                        fixed_variance=hpp.trend_exp_fixed_variance,
                        num_samples=num_samples,device=device)
        )
    result.offset_lin = Uniform(hpp.offset_lin_min, hpp.offset_lin_max).sample([num_samples]).to(device)
    result.offset_exp = Uniform(hpp.offset_exp_min, hpp.offset_exp_max).sample([num_samples]).to(device)

    result.noise_k = double_sampling(min_val=hpp.noise_k_min,max_val=hpp.noise_k_max,
                                     num_samples=num_samples,device=device)
    result.noise_scale = noise_scale_sampling(num_samples, device=device)

    result.amplitude = Uniform(hpp.amplitude_min, hpp.amplitude_max).sample([num_samples]).to(device)
    result.p_add = torch.rand(num_samples, device=device) < hpp.p_add
    result.trend_damp = Uniform(hpp.trend_damp_min,hpp.trend_damp_max).sample([num_samples]).to(device)
    
    result.p_noise_trend_scaling = torch.rand(num_samples, device=device) < hpp.p_noise_trend_scaling
    result.p_strong_exp = torch.rand(num_samples, device=device) < hpp.p_strong_exp

    # keep the n-days at a set median
    mm = hpp.resolution_multiplier
    f_res = lambda x: torch.log2(x * mm + 1)
    f_res_inv = lambda x: (2**x - 1) / mm

    result.resolution = f_res_inv(Uniform(f_res(torch.scalar_tensor(hpp.resolution_min)),
                                          f_res(torch.scalar_tensor(hpp.resolution_max)))
                                  .sample()).to(device).repeat(num_samples)
    

    return result



def double_sampling(min_val, max_val, num_samples=1, beta_a=2, beta_b=2, device="cpu"):
    z = Beta(beta_a, beta_b).sample((num_samples,)).to(device) # Beta(2,2) -> [0,1]
    return min_val + (max_val - min_val) * z        # scale to [min,max]

def triple_sampling(min_val, max_val, fixed_variance, num_samples=1, beta_a=2, beta_b=2, device="cpu"):
    z = double_sampling(min_val, max_val, num_samples, beta_a, beta_b, device)
    gaus_noise = torch.randn(num_samples, device=device) * fixed_variance  # N(0,sigma^2)
    return z + gaus_noise

def noise_scale_sampling(num_samples: int, low_prob=0.4, med_prob=0.8,
                         device: str = "cpu"):
    rand = np.random.rand()
    if rand <= low_prob: noise = Uniform(0, 0.2).sample([num_samples])      # very low noise
    elif rand <= med_prob: noise = Uniform(0.3, 0.7).sample([num_samples])  # moderate noise
    else: noise = Uniform(0.8, 1.2).sample([num_samples])                   # high noise

    return noise.to(device)