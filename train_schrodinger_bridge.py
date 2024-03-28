from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime as dt
import torch

import colored_traceback.always

from runner import Runner
import util
import options


print("="*80)
print(f"\t\tTraining start at {dt.datetime.now().strftime('%m_%d_%Y_%H%M%S')}")
print("="*80)
print("setting configurations...")
opt = options.set("parametric")

def main(opt):
    run = Runner(opt)

    if opt.train_method == 'alternate':
        run.sb_alternate_train(opt)

    elif opt.train_method in ['alternate_imputation', 'alternate_imputation_v2']:
        run.sb_alternate_imputation_train(opt)

    elif opt.train_method == 'alternate_backward':
        run.sb_alternate_train_backward(opt)

    elif opt.train_method in ['alternate_backward_imputation', 'alternate_backward_imputation_v2']:
        run.sb_alternate_imputation_train_backward(opt)

    elif opt.train_method in ['dsm', 'dsm_v2', 'dsm_imputation', 'dsm_imputation_v2',
        'dsm_imputation_forward_verfication']:
        run.sb_dsm_train(opt)
    
    elif opt.train_method in ['mdp_train']:
        run.sb_alternate_imputation_train_mdp(opt)

    else:
        raise NotImplementedError(f'New train method {opt.train_method}')
    
    run.parametric_sampling(opt)

if not opt.cpu:
    with torch.cuda.device(opt.gpu):
        main(opt)
else: main(opt)
