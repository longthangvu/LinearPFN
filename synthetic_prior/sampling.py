import torch, random
import numpy as np
from torch.distributions import Beta, Uniform

from synthetic_prior.misc import dotdict

def sample_from_hyperpriors(hpp, n_context=2, n_sequence=12000, device="cpu"):
    result = dotdict()

    for param, min_val, max_val, fixed_variance in [
        ("annual_param", hpp.a_min, hpp.a_max, hpp.a_fixed_variance),
        ("monthly_param", hpp.m_min, hpp.m_max, hpp.m_fixed_variance),
        ("weekly_param", hpp.w_min, hpp.w_max, hpp.w_fixed_variance),
        ("frequency_zero_inflation", hpp.f_zi_min, hpp.f_zi_max, hpp.f_zi_fixed_variance),  # noqa
        ("trend_lin", hpp.trend_lin_min, hpp.trend_lin_max, hpp.trend_lin_fixed_variance),  # noqa
    ]:
        result[param] = triple_sampling(min_val,max_val,
                                        fixed_variance=fixed_variance,
                                        num_samples=n_context,device=device)

    # make it equally likely to have a positive or negative exp trend

    mm = hpp.trend_exp_multiplier  
    # f_exp = lambda x: 2 ** ((x - 1) * mm)
    # f_exp_inv = lambda x: (torch.log2(x) / mm) + 1
    
    # shrink effective multiplier as series span grows to keep base^(time) bounded
    # approximate upper span by n_sequence / resolution_min
    span_upper = n_sequence / hpp.resolution_min
    mm_eff = hpp.trend_exp_multiplier / span_upper
    f_exp      = lambda x: 2 ** ((x - 1) * mm_eff)
    f_exp_inv  = lambda x: (torch.log2(x) / mm_eff) + 1


    result.trend_exp = f_exp_inv(
        triple_sampling(f_exp(torch.scalar_tensor(hpp.trend_exp_min)),
                        f_exp(torch.scalar_tensor(hpp.trend_exp_max)),
                        fixed_variance=hpp.trend_exp_fixed_variance,
                        num_samples=n_context,device=device)
        )

    # ensure consistent sign for trends

    median_lin_sign = result.trend_lin.median().sign()
    result.trend_lin = result.trend_lin.abs() * median_lin_sign

    assert (result.trend_lin >= 0).all() or (
        result.trend_lin <= 0
    ).all(), f"non-consistent sign {result.trend_lin=} in trend_lin"

    median_exp_sign = (result.trend_exp - 1).median().sign()
    result.trend_exp = (result.trend_exp - 1).abs() * median_exp_sign + 1

    assert (result.trend_exp >= 1).all() or (
        result.trend_exp <= 1
    ).all(), f"non-consistent {result.trend_exp=} in trend_exp"

    # sub-context-specific params

    result.noise_k = double_sampling(hpp.noise_k_min,hpp.noise_k_max,
                                        num_samples=n_context,device=device)
    result.noise_scale = noise_scale_sampling(n_context, device=device)

    # domain-specific params

    result.discreteness = (
        Uniform(hpp.discreteness_min, hpp.discreteness_max)
        .sample([n_context])
        .to(device)
    )

    result.bias_zi = (
        Uniform(hpp.bias_zi_min, hpp.bias_zi_max).sample([n_context]).to(device)
    )

    result.amplitude = (
        Uniform(hpp.amplitude_min, hpp.amplitude_max).sample([n_context]).to(device)
    )

    # result.non_negative = (
    #     Categorical(
    #         torch.tensor([1 - hpp.non_negative_prob, hpp.non_negative_prob])
    #     )
    #     .sample()
    #     .to(device)
    #     .repeat(n_context)
    # )

    result.offset_lin = (
        Uniform(hpp.offset_lin_min, hpp.offset_lin_max)
        .sample([n_context])
        .to(device)
    )

    result.offset_exp = (
        Uniform(hpp.offset_exp_min, hpp.offset_exp_max)
        .sample([n_context])
        .to(device)
    )

    result.harmonics = torch.randint(hpp.harmonics_min, hpp.harmonics_max, (3,)).to(
        device
    )

    # keep the n-days at a set median

    mm = hpp.resolution_multiplier
    f_res = lambda x: torch.log2(x * mm + 1)
    f_res_inv = lambda x: (2**x - 1) / mm

    result.resolution = (
        f_res_inv(
            Uniform(
                f_res(torch.scalar_tensor(hpp.resolution_min)),
                f_res(torch.scalar_tensor(hpp.resolution_max)),
            ).sample()
        )
        .to(device)
        .repeat(n_context)
    )

    result.n_units = torch.ceil(n_sequence / result.resolution)
    result.p_add = random.random() < hpp.p_add
    result.trend_damp = Uniform(hpp.trend_damp_min,hpp.trend_damp_max).sample()

    return result



def double_sampling(min_val, max_val, num_samples=1, beta_a=2, beta_b=2, device="cpu"):
    z = Beta(beta_a, beta_b).sample((num_samples,)).to(device) # Beta(2,2) -> [0,1]
    return min_val + (max_val - min_val) * z        # scale to [min,max]

def triple_sampling(min_val, max_val, fixed_variance, num_samples=1, beta_a=2, beta_b=2, device="cpu"):
    z = double_sampling(min_val, max_val, num_samples, beta_a, beta_b, device)
    gaus_noise = torch.randn(num_samples, device=device) * fixed_variance  # N(0,sigma^2)
    return z + gaus_noise

def noise_scale_sampling(num_samples: int, device: str = "cpu"):
    rand = np.random.rand()
    # very low noise
    if rand <= 0.4:
        noise = Uniform(0, 0.2).sample([num_samples])
    # moderate noise
    elif rand <= 0.8:
        noise = Uniform(0.3, 0.7).sample([num_samples])
    # high noise
    else:
        noise = Uniform(0.8, 1.2).sample([num_samples])

    return noise.to(device)
# def noise_scale_sampling(num_samples: int, low_prob=0.4, med_prob=0.8,
#                          device: str = "cpu"):
#     rand = np.random.rand()
#     if rand <= low_prob: noise = Uniform(0, 0.2).sample([num_samples])      # very low noise
#     elif rand <= med_prob: noise = Uniform(0.3, 0.7).sample([num_samples])  # moderate noise
#     else: noise = Uniform(0.8, 1.2).sample([num_samples])                   # high noise

#     return noise.to(device)