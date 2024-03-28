import numpy as np

import torch
import torch.distributions as td
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from ipdb import set_trace as debug

from scipy.interpolate import InterpolatedUnivariateSpline
from prefetch_generator import BackgroundGenerator
import util
import tensorflow as tf

NUM_WORKERS = 0

def build_boundary_distribution(opt, behavior_dataset=None):
    print("build boundary distribution...")

    # opt.data_dim = get_data_dim()
    pdata = build_data_sampler(opt, opt.samp_bs, behavior_dataset)
    prior = build_prior_sampler(opt, opt.samp_bs)
    

    return pdata, prior

def get_data_dim():
    return [1, 1, 2]

def build_prior_sampler(opt, batch_size):
    if opt.problem_name == 'moon-to-spiral':
        # 'moon-to-spiral' uses Moon as prior distribution
        return Moon(batch_size)

    # image+VESDE -> use (sigma_max)^2; otherwise use 1.
    # cov_coef = opt.sigma_max**2 if (util.is_image_dataset(opt) and not util.use_vp_sde(opt)) else 1.
    if util.is_image_dataset(opt) and not util.use_vp_sde(opt):
        cov_coef = opt.sigma_max**2
    elif opt.sde_type == 've' and opt.problem_name == 'gmm':
        cov_coef = opt.sigma_max**2 + 7**2  # mimic the forward marginal prior distribution.
    elif opt.sde_type == 've':
        cov_coef = opt.sigma_max**2
    else:
        cov_coef = 1

    print('prior cov', cov_coef)
    prior = td.MultivariateNormal(torch.zeros(opt.data_dim, device=opt.device),
                                  cov_coef*torch.eye(opt.data_dim[-1], device=opt.device))
    return PriorSampler(prior, batch_size, opt.device)

def build_data_sampler(opt, batch_size, behavior_dataset=None):
    if opt.problem_name == "mdp":
        dataset_generator = generate_mdp_dataset
        dataset = dataset_generator(opt, behavior_dataset)
    elif opt.problem_name == "parametric":
        dataset_generator = generate_mdp_dataset
        dataset = dataset_generator(opt, behavior_dataset)
    elif opt.problem_name == 'pm25':
        dataset_generator = generate_pm25_dataset
        dataset = dataset_generator(opt)
    return DataSampler(dataset, batch_size, opt.device, opt=opt, torch_loader=False)


class MixMultiVariateNormal:
    def __init__(self, batch_size, radius=12, num=8, sigmas=None, device='cpu'):
        # build mu's and sigma's. num: number of modes.
        arc = 2*np.pi/num
        xs = [np.cos(arc*idx)*radius for idx in range(num)]
        ys = [np.sin(arc*idx)*radius for idx in range(num)]
        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        dim = len(mus[0])
        sigmas = [torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        print('batch_size', batch_size, 'num', num)
        if batch_size%num!=0:
            raise ValueError(f'batch size {batch_size} must be devided by number of gaussian {num}')
        self.num = num
        self.batch_size = batch_size
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]
        self.device = device

    def log_prob(self,x):
        # assume equally-weighted
        densities=[torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities)/len(self.dists))

    def sample(self, num_samples=None):
        if num_samples is None:
            num_samples = self.batch_size
        # build mu's and sigma's. num: number of modes.
        ind_sample = num_samples / self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        return samples.to(self.device)

class CheckerBoard:
    def __init__(self, batch_size, device='cpu'):
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        n = self.batch_size
        n_points = 3*n
        n_classes = 2
        freq = 5
        x = np.random.uniform(-(freq//2)*np.pi, (freq//2)*np.pi, size=(n_points, n_classes))
        mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
        np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0))
        y = np.eye(n_classes)[1*mask]
        x0=x[:,0]*y[:,0]
        x1=x[:,1]*y[:,0]
        sample=np.concatenate([x0[...,None],x1[...,None]],axis=-1)
        sqr=np.sum(np.square(sample),axis=-1)
        idxs=np.where(sqr==0)
        sample=np.delete(sample,idxs,axis=0)
        # res=res+np.random.randn(*res.shape)*1
        sample=torch.Tensor(sample)
        sample=sample[0:n,:]
        return sample

class Spiral:
    def __init__(self, batch_size, device='cpu'):
        self.batch_size = batch_size
        self.device = device

    def sample(self, num_samples=None):
        n = self.batch_size if num_samples is None else num_samples
        theta = np.sqrt(np.random.rand(n))*3*np.pi-0.5*np.pi # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 0.25*np.random.randn(n,2)
        samples = np.append(x_a, np.zeros((n,1)), axis=1)
        samples = samples[:,0:2]
        return torch.Tensor(samples).to(self.device)

class Moon:
    def __init__(self, batch_size, device='cpu'):
        self.batch_size = batch_size
        self.device = device

    def sample(self, num_samples=None):
        n = self.batch_size if num_samples is None else num_samples
        x = np.linspace(0, np.pi, n // 2)
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1) * 10.
        u += 0.5*np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1) * 10.
        v += 0.5*np.random.normal(size=v.shape)
        x = np.concatenate([u, v], axis=0)
        return torch.Tensor(x).to(self.device)

class DataSampler: # a dump data sampler
    def __init__(self, dataset=None, batch_size=32, device='cpu', opt=None, torch_loader=False):
        self.num_sample = len(dataset)

        self.dataloader = setup_loader(dataset, batch_size, torch_loader=torch_loader)
        self.batch_size = batch_size
        self.device = device
        self.opt = opt

    def sample(self, num_samples=None, return_mask=False, return_all_mask=False):
        # need batch_permute
        # if self.opt.problem_name in ['sinusoid', 'pm25', 'physio',
        #     'exchange_rate_nips', 'solar_nips', 'electricity_nips',]:
        if self.opt.problem_name in ['sinusoid', 'pm25', 'physio',
            'exchange_rate_nips', 'solar_nips', 'electricity_nips',"mdp", "parametric"]:
            assert self.opt.permute_batch == True

            data = next(iter(self.dataloader))
            obs_data = data['observed_data'].float().permute(0,2,1)  # (B,L,K).permute->(B,K,L)

            if return_all_mask:
            # obs is condition gt is target

                obs_mask = data['observed_mask'].float().permute(0,2,1)  # (B,L,K) (B,K,L)
                obs_data = obs_data * obs_mask
                gt_mask = data["gt_mask"].float().permute(0,2,1)
                return (obs_data.unsqueeze(1).to(self.device),
                        obs_mask.unsqueeze(1).to(self.device),
                        gt_mask.unsqueeze(1).to(self.device))

            elif return_mask:
                obs_mask = data['observed_mask'].float().permute(0,2,1)  # (B,L,K) (B,K,L)
                obs_data = obs_data * obs_mask
                # if hasattr(self.opt, 'interpolate') and self.opt.interpolate:
                #     obs_data = interpolate(obs_data, obs_mask)  # x, mask (B,K,L)
                return (obs_data.unsqueeze(1).to(self.device),
                        obs_mask.unsqueeze(1).to(self.device))
            else:
                return obs_data.unsqueeze(1).to(self.device)

        elif self.opt.problem_name in ['gmm', 'checkerboard', 'moon-to-spiral']:
            # data = next(self.dataloader)  # original code using yield.
            data = next(iter(self.dataloader))
            return data[0].to(self.device)

        elif self.opt.problem_name in ['mdp']:
            data = next(iter(self.dataloader))
            obs_data = data['observed_data'].float().permute(0,2,1)  # (B,L,K).permute->(B,K,L)
            cond_mask = data['cond_mask'].float().permute(0,2,1)  # (B,L,K).permute->(B,K,L)
            if return_mask or return_all_mask:
                return (obs_data.unsqueeze(1).to(self.device),
                    cond_mask.unsqueeze(1).to(self.device))
            else:
                return obs_data.unsqueeze(1).to(self.device)
        
        else:
            raise NotImplementedError(f'Dataset {self.opt.problem_name} is unknown.')


def interpolate(observed_data, cond_mask):
    B, K, L = observed_data.shape
    cond_obs = cond_mask * observed_data
    timeline = np.arange(L)
    cond_mask = cond_mask.cpu().numpy()
    imputed_samples = cond_obs.clone().cpu().numpy()
    interpolate_order = 1

    for b in range(B):
        for k in range(K):
            obs_points = cond_mask[b,k] == 1
            target_points = ~obs_points
            if np.sum(obs_points) == L:
                continue
            obs_x = timeline[obs_points]
            obs_y = observed_data[b,k,obs_points].cpu().numpy()
            if np.sum(obs_points) > interpolate_order:  # Empty feature.
                target_x = timeline[target_points]
                target_x_clipped = np.clip(target_x, a_min=obs_x[0], a_max=obs_x[-1])
                interpolate_func = InterpolatedUnivariateSpline(obs_x, obs_y, k=interpolate_order)
                inpute_y = interpolate_func(target_x_clipped)
            else:
                num_obs = cond_mask[b].sum(axis=0)  # L
                num_obs[num_obs == 0] = 1
                obs_y_mean = cond_obs[b].cpu().numpy().sum(axis=0) / num_obs  # L, Mean over K.
                inpute_y = obs_y_mean[target_points]

            imputed_samples[b,k,target_points] = inpute_y

    return torch.from_numpy(imputed_samples)

class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, device):
        self.prior = prior
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self, num_samples=None):
        num_samples = self.batch_size if num_samples is None else num_samples
        return self.prior.sample([num_samples]).to(self.device)



def setup_loader(dataset=None, batch_size=32, torch_loader=False):

    if not torch_loader:
        # train_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True,
        #      num_workers=NUM_WORKERS, drop_last=True)
        train_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, drop_last=True)
        # train_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True,
        #     num_workers=NUM_WORKERS, pin_memory=True)
        # print("number of samples: {}".format(len(dataset)))
        # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py#L52-L53
        # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/train_util.py#L166
        while True:
            yield from train_loader
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=False)
        while True:
            yield from train_loader
        # return train_loader

class  Mdp_Dataset():
    def __init__(
            self,
            eval_length=48,
            target_dim=35,
            seed=0,
            behavior_dataset=None):
        self.eval_length = eval_length
        self.target_dim = target_dim
        np.random.seed(seed)  # seed for ground truth choice

        state_dim = behavior_dataset.states.shape[1]
        action_dim = behavior_dataset.actions.shape[1]
        self.obs = np.concatenate((behavior_dataset.states, behavior_dataset.actions, behavior_dataset.next_states, behavior_dataset.rewards), axis = 1)
        self.obs = self.obs[:,:,np.newaxis]
        self.obs_mask = np.ones_like(self.obs)
        self.gt_mask = np.zeros_like(self.obs)
        self.gt_mask[:, : state_dim + action_dim, 0] = 1


    def __getitem__(self, org_index):
        
        s = {
            'observed_data':self.obs[org_index],
            'observed_mask':self.obs_mask[org_index]
            ,"gt_mask":self.gt_mask[org_index]
        }
        return s

    def __len__(self):
        return self.obs.shape[0]

 
class parametric_dataset():
    def __init__(self, opt):
        size = 10000
        x = np.random.uniform(opt.a, opt.b, size = size)

        sigma = opt.sigma
        try:
            noise = np.random.normal(opt.mu, sigma, size = size)
        except:
            noise = opt.mu * np.ones(size)
        y = x + noise
        self.obs = np.column_stack((x,y))[:,:,np.newaxis]
        self.obs_mask = np.column_stack((np.ones(size), np.ones(size)))[:,:,np.newaxis]
        self.gt_mask = np.column_stack((np.ones(size), np.zeros(size)))[:,:,np.newaxis]
        
    
    def __getitem__(self, org_index):
        
        s = {
            "observed_data":self.obs[org_index],
            "observed_mask":self.obs_mask[org_index]
            ,"gt_mask":self.gt_mask[org_index]
        }
        return s
    
    def __len__(self):
        return self.obs.shape[0]


def get_dataloader(
        opt,
        seed=1,
        # nfold=0,
        # batch_size=16,
        # eval_length=48,
        # missing_ratio=0.1,
        # device='cpu',
        # return_dataset=False,
        behavior_dataset=None):
    """Create dataloaders."""

    # only to obtain total length of dataset
    np.random.seed(seed)
    if opt.problem_name == "mdp":
        dataset = Mdp_Dataset(seed=seed,behavior_dataset=behavior_dataset)
    elif opt.problem_name == "parametric":
        dataset = parametric_dataset(opt)
    
    # we can here set all the dimension parameter in opt
    opt.data_dim = [1, dataset.obs.shape[2], dataset.obs.shape[1]]
    opt.model_configs[opt.forward_net].input_size = (1, dataset.obs.shape[1])
    opt.model_configs[opt.backward_net].input_size = (1, dataset.obs.shape[1])
    opt.input_size = (1, dataset.obs.shape[1])

    print('train/test/val num samples', len(dataset))
    return dataset





class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def generate_mdp_dataset(opt, behavior_dataset):
    # opt.data_dim = behavior_dataset.states.shape[0]
    train_dataset = get_dataloader(opt,
        seed=1, behavior_dataset=behavior_dataset)
    # print('num samples', len(train_dataset))
    return train_dataset

