import os, json
import torch

def build_model(model_id, model_name='LinearPFN', ckpts_root='training/ckpts', ckpt_file='best_model.pt'):
    if model_name == 'LinearPFN':
        from training.models.LinearPFN import LinearPFN
        mdl = LinearPFN

    with open(f'{ckpts_root}/{model_name}/{model_id}/run_config.json', 'r') as file:
        model_args = json.load(file)['model_params']
    model = mdl(**model_args)
    
    print('Loading model from checkpoint')
    model_folder = f'{ckpts_root}/{model_name}/{model_id}/'
    ckpt = torch.load(os.path.join(model_folder, ckpt_file), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    return model