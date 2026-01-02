import os, argparse
import numpy as np
import torch, random
import torch.backends

from exp.exp_meta_learning_pfn import Exp_MetaLearningPFN

from utils.print_args import print_args


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Running Experiment')
    
    # parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
    #                     help='task name, options:[long_term_forecast, short_term_forecast]')
    # parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='LinearPFN',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../Time-Series-Library/dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # additional params for pfns
    parser.add_argument('--train_budget', type=float, default=1.0, help='length of training sequence')
    parser.add_argument('--ckpts_root', type=str, default='training/ckpts', help='location of model checkpoints')
    parser.add_argument('--ckpt_file', type=str, default='best_model.pt', help='model checkpoints filename')
    # parser.add_argument('--data_version', type=str, default='')
    # parser.add_argument('--c_min', type=int, default=32)
    # parser.add_argument('--c_max', type=int, default=1536)
    # parser.add_argument('--train_stride', type=int, default=16)
    # parser.add_argument('--use_time', type=bool, default=True)
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # # model define
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    # parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    # parser.add_argument('--factor', type=int, default=1, help='attn factor')
    # parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    # parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    # parser.add_argument('--c_out', type=int, default=7, help='output size')
    # parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    # parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    # parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    # parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    # parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # parser.add_argument('--activation', type=str, default='gelu', help='activation')
    # parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    # parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    # parser.add_argument('--distil', action='store_false',
    #                     help='whether to use distilling in encoder, using this argument means not using distilling',
    #                     default=True)
    # parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    # parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    # # de-stationary projector params
    # parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
    #                     help='hidden layer dimensions of projector (List)')
    # parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # # TimeMixer params
    # parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    # parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    # parser.add_argument('--down_sampling_method', type=str, default=None,
    #                     help='down sampling method, only support avg, max, conv')
    # parser.add_argument('--channel_independence', type=int, default=1,
    #                     help='0: channel dependence 1: channel independence for FreTS model')
    # parser.add_argument('--decomp_method', type=str, default='moving_avg',
    #                     help='method of series decompsition, only support moving_avg or dft_decomp')
    # parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    # # metrics (dtw)
    # parser.add_argument('--use_dtw', type=bool, default=False,
    #                     help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
 
    # # augmentation
    # parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    
    # # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # parser.add_argument('--itr', type=int, default=1, help='experiments times')
    # parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--des', type=str, default='test', help='exp description')
    # parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # print('Args in experiment:')
    # print_args(args)

    exp = Exp_MetaLearningPFN(args)  # set experiments

    setting = f'{args.model}/{args.model_id}'
    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    exp.test(setting, test=1)
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()