import os, json
import torch

def build_model(model_id, model_name='LinearPFN', ckpts_root='output', ckpt_file='best_model.pt'):
    if model_name == 'LinearPFN':
        from training.models.LinearPFN import LinearPFN
        mdl = LinearPFN

    ckpt_dir=f'{ckpts_root}/{model_name}/{model_id}'
    assert os.path.exists(ckpt_dir), "Checkpoints directory doesn't exist"
    with open(f'{ckpt_dir}/run_config.json', 'r') as file:
        model_args = json.load(file)['model_params']
    model = mdl(**model_args)
    
    print('Loading model from checkpoint...')
    ckpt_path = os.path.join(f'{ckpt_dir}/ckpts', ckpt_file)
    print(ckpt_path)
    print(model_args)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    return model_args, model