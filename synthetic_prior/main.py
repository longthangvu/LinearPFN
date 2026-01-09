import argparse, json, math
import numpy as np
from pathlib import Path
from tqdm import tqdm

from synthetic_prior.sampling import sample_from_hyperpriors
from synthetic_prior.misc import dotdict
from synthetic_prior.generation import generate_series

def main(args):
    out_dir = f'series_bank/{args.dataset_id}'
    with open(f'{out_dir}/meta.json', 'r') as file:
        meta_data = json.load(file)

    hpp = dotdict(meta_data['hyperprior_params'])

    num_series, series_len = meta_data['num_series'], meta_data['series_len']
    shard_sz, batch_sz = meta_data['shard_size'], meta_data['batch_size']
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_shards = math.ceil(num_series / shard_sz)
    global_idx = 0

    for s in tqdm(range(num_shards), desc="shards"):
        k = min(shard_sz, num_series - global_idx)
        # buf_y, buf_noise = np.empty((k, series_len)), np.empty((k, series_len))
        buf_y = np.empty((k, series_len))

        write_ptr = 0

        while write_ptr < k:
            b = min(batch_sz, k - write_ptr)
            comp_params = sample_from_hyperpriors(hpp, series_len=series_len, num_samples=b)
            y, _ = generate_series(comp_params, series_len=series_len, num_samples=b)
            # y, noise: [b, 12000]

            y_np     = y[:b].contiguous().cpu().numpy()
            # noise_np = noise[:b].contiguous().cpu().numpy()
            buf_y[write_ptr : write_ptr + b]     = y_np
            # buf_noise[write_ptr : write_ptr + b] = noise_np

            write_ptr += b

        np.save(out_dir / f"y_shard_{s:04d}.npy", buf_y)
        # np.save(out_dir / f"noise_shard_{s:04d}.npy", buf_noise)

        global_idx += k

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic Data Generation Script')
    parser.add_argument('--dataset_id', type=str, default='exp', help='dataset output directory')

    args = parser.parse_args()
    main(args)