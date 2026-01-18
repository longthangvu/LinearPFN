import os, argparse, time, json
import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from utils.build_model import build_model

from training.data.meta_dataset import VariableMetaDataset
from training.util import validate_model, gaussian_nll_loss, get_opt_lr_schedule
from utils.tools import set_seed

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # model
    ckpts_root = args.ckpts_root
    model_name, model_id = args.model, args.model_id
    ckpt_dir=f'{ckpts_root}/{model_name}/{model_id}'
    assert os.path.exists(ckpt_dir), "Checkpoints directory doesn't exist"
    with open(f'{ckpt_dir}/run_config.json', 'r') as file:
        run_config = json.load(file)
        model_args = run_config['model_params']
    model = build_model(model_args, model_name).to(device)

    # dataset
    dataset = VariableMetaDataset(**run_config['data'], device=device)
    dataset_meta = f"C{dataset.C_range}_Q{dataset.Q_range}"
    
    # train config
    save_dir = f"{ckpt_dir}/{args.seed}/ckpts"
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    log_dir = f"tb_train/{model_name}/{model_id}/{args.seed}/{dataset_meta}"
    writer = SummaryWriter(log_dir=log_dir)
    
    config = run_config['train_config']
    optimizer, scheduler = get_opt_lr_schedule(model, config)
    criterion = nn.MSELoss()
    n_epochs = config['total_tasks'] // config['tasks_per_epoch']
    
    print(f"\nTraining Plan:")
    print(f"  Total meta-tasks: {config['total_tasks']:,}")
    print(f"  Tasks per epoch: {config['tasks_per_epoch']:,}")
    print(f"  Total epochs: {n_epochs}")

    # init
    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    global_task_count = 0
    
    print("Running initial validation...")
    val_metrics = validate_model(model, dataset, device, config['n_val_tasks'])
    print(f"Initial validation - Loss: {val_metrics['loss']:.4f}, Corr: {val_metrics['correlation']:.3f}")
    writer.add_scalar('val/loss', val_metrics['loss'], 0)
    writer.add_scalar('val/loss_std', val_metrics['loss_std'], 0)
    writer.add_scalar('val/corr', val_metrics['correlation'], 0)
    writer.add_scalar('val/corr_std', val_metrics['correlation_std'], 0)

    model.eval()
    with torch.no_grad():
        eg = dataset.create_meta_task()
        eg_ctx_x = eg['ctx_x'].unsqueeze(0).float().to(device)
        eg_ctx_z = eg['ctx_z'].unsqueeze(0).float().to(device)
        eg_qry_x = eg['qry_x'].unsqueeze(0).float().to(device)
        eg_t_ctx = torch.tensor(eg['endpoints']['ctx'], dtype=torch.long).unsqueeze(0).to(device)
        eg_t_qry = torch.tensor(eg['endpoints']['qry'], dtype=torch.long).unsqueeze(0).to(device)

        try:
            writer.add_graph(model, (eg_ctx_x, eg_ctx_z, eg_qry_x, eg_t_ctx, eg_t_qry))
        except Exception:
            traced = torch.jit.trace(
                model,
                (eg_ctx_x, eg_ctx_z, eg_qry_x, eg_t_ctx, eg_t_qry),
                strict=False
            )
            writer.add_graph(traced, (eg_ctx_x, eg_ctx_z, eg_qry_x, eg_t_ctx, eg_t_qry))

    # training loop
    model.train()
    global_step = 0

    for epoch in tqdm(range(n_epochs)):
        epoch_start = time.time()
        epoch_losses = []
        
        for task_idx in range(config['tasks_per_epoch']):
            optimizer.zero_grad()
            task = dataset.create_meta_task()
            ctx_x = task['ctx_x'].unsqueeze(0).float().to(device)
            ctx_z = task['ctx_z'].unsqueeze(0).float().to(device)
            qry_x = task['qry_x'].unsqueeze(0).float().to(device)
            qry_z = task['qry_z'].unsqueeze(0).float().to(device)
            t_ctx = torch.tensor(task['endpoints']['ctx'], dtype=torch.long).unsqueeze(0).to(device)  # [1,C]
            t_qry = torch.tensor(task['endpoints']['qry'], dtype=torch.long).unsqueeze(0).to(device)  # [1,Q]

            pred = model(ctx_x, ctx_z, qry_x, t_ctx, t_qry)
            loss = criterion(pred, qry_z)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            global_task_count += 1
            global_step += 1

            if task_idx % config['log_every'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                elapsed_hours = (time.time() - start_time) / 3600
                tasks_per_hour = global_task_count / max(elapsed_hours, 1e-9)

                print(f"Epoch {epoch+1:3d}/{n_epochs}, Task {global_task_count:6,}/{config['total_tasks']:,}: "
                    f"Loss={loss.item():.4f}, LR={current_lr:.2e}, t={elapsed_hours:.2f}h")

                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('opt/lr', current_lr, global_step)
                writer.add_scalar('speed/tasks_per_hour', tasks_per_hour, global_step)

        avg_train_loss = float(np.mean(epoch_losses))
        
        history['loss'].append(avg_train_loss)
        history['epoch'].append(epoch + 1)
        

        epoch_time = time.time() - epoch_start

        writer.add_scalar('train/epoch_loss', avg_train_loss, epoch + 1)
        writer.add_scalar('time/epoch_seconds', epoch_time, epoch + 1)
        
        # Validation
        if (epoch + 1) % config['validate_every'] == 0:
            print(f"\nValidation after epoch {epoch + 1}...")
            val_metrics = validate_model(model, dataset, device, config['n_val_tasks'])
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_correlation'].append(val_metrics['correlation'])
            history['val_epoch'].append(epoch + 1)
            
            tasks_per_hour = global_task_count / ((time.time() - start_time) / 3600)
            
            print(f"Epoch {epoch+1:3d} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} ± {val_metrics['loss_std']:.4f}")
            print(f"  Val Corr: {val_metrics['correlation']:.3f} ± {val_metrics['correlation_std']:.3f}")
            print(f"  Speed: {tasks_per_hour:,.0f} tasks/hour, Tasks: {global_task_count:,}")

            writer.add_scalar('val/loss', val_metrics['loss'], epoch + 1)
            writer.add_scalar('val/loss_std', val_metrics['loss_std'], epoch + 1)
            writer.add_scalar('val/corr', val_metrics['correlation'], epoch + 1)
            writer.add_scalar('val/corr_std', val_metrics['correlation_std'], epoch + 1)

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_correlation': val_metrics['correlation'],
                    'epoch': epoch + 1,
                    'global_tasks': global_task_count,
                    'config': config,
                    'dataset_config': {
                        'L': dataset.L, 'H': dataset.H,
                        'C_range': dataset.C_range, 'Q_range': dataset.Q_range
                    }
                }, f'{save_dir}/best_model.pt')
                print(f"Best model saved (tasks: {global_task_count:,})")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping after {patience_counter} epochs without improvement")
                writer.flush()
                writer.close()
                break
        
        # Regular checkpoints
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'global_tasks': global_task_count,
                'history': dict(history)
            }, f'{save_dir}/model_epoch_{epoch+1}.pt')
            writer.flush()

    writer.flush()
    writer.close()

    total_time = time.time() - start_time
    tasks_per_hour = global_task_count / (total_time / 3600)

    print(f"\n TRAINING COMPLETE! ")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total tasks trained: {global_task_count:,}")
    print(f"Average speed: {tasks_per_hour:,.0f} tasks/hour")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved in {save_dir}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--ckpts_root', type=str, default='output', help='output directory')
    parser.add_argument('--model', type=str, default='LinearPFN', help='model name')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--seed', type=int, default=42, help='model seed')
    args = parser.parse_args()
    
    set_seed(args.seed)
    # print(args.seed)
    main(args)