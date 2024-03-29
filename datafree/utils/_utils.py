from curses import KEY_SCREATE
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset
import numpy as np 
from PIL import Image
import os, random, math
from copy import deepcopy
from contextlib import contextmanager
import torch.nn.functional as F
import torch.distributed as dist

def estimate_gradient_objective(victim_model, clone_model, x, epsilon = 1e-7, m = 5, verb=False, num_classes=10, device = "cpu", pre_x=False, forward_differences=True, no_logits=1, loss='l1', logit_correction='mean'):
    # Sampling from unit sphere is the method 3 from this website:
    #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    #x = torch.Tensor(np.arange(2*1*7*7).reshape(-1, 1, 7, 7))
    
    # if pre_x and args.G_activation is None:
    #     raise ValueError(args.G_activation)

    clone_model.eval()
    victim_model.eval()
    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)
        C = x.size(1)
        S = x.size(2)
        dim = S**2 * C

        u = np.random.randn(N * m * dim).reshape(-1, m, dim) # generate random points from normal distribution

        d = np.sqrt(np.sum(u ** 2, axis = 2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim = 1) # Shape N, m + 1, S^2

            

        u = u.view(-1, m + 1, C, S, S)

        evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
        if pre_x: 
            evaluation_points = torch.tanh(evaluation_points) # Apply args.G_activation function

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_clone = []
        max_number_points = 32*156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU
        
        for i in (range(N * m // max_number_points + 1)): 
            pts = evaluation_points[i * max_number_points: (i+1) * max_number_points]
            pts = pts.to(device)

            pred_victim_pts = victim_model(pts).detach()
            pred_clone_pts = clone_model(pts)

            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)



        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_clone = torch.cat(pred_clone, dim=0).to(device)

        u = u.to(device)

        if loss == "l1":
            loss_fn = F.l1_loss
            if no_logits:
                pred_victim = F.log_softmax(pred_victim, dim=1).detach()
                if logit_correction == 'min':
                    pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()
                elif logit_correction == 'mean':
                    pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()


        elif loss == "kl":
            loss_fn = F.kl_div
            pred_clone = F.log_softmax(pred_clone, dim=1)
            pred_victim = F.softmax(pred_victim.detach(), dim=1)

        else:
            raise ValueError(loss)

        # Compute loss
        if loss == "kl":
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').sum(dim = 1).view(-1, m + 1) 
        else:
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').mean(dim = 1).view(-1, m + 1) 

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        if forward_differences:
            gradient_estimates *= dim            

        if loss == "kl":
            gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, S, S) 
        else:
            gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, S, S) / (num_classes * N) 

        clone_model.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


def compute_gradient(args, victim_model, clone_model, x, pre_x=False, device="cpu"):
    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    clone_model.eval()
    N = x.size(0)
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)


    if pre_x:
        x_ = args.G_activation(x_)


    pred_victim = victim_model(x_)
    pred_clone = clone_model(x_)

    if args.loss == "l1":
        loss_fn = F.l1_loss
        if args.no_logits:
            pred_victim_no_logits = F.log_softmax(pred_victim, dim=1)
            if args.logit_correction == 'min':
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.min(dim=1).values.view(-1, 1)
            elif args.logit_correction == 'mean':
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.mean(dim=1).view(-1, 1)
            else:
                pred_victim = pred_victim_no_logits

    elif args.loss == "kl":
        loss_fn = F.kl_div
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_victim = F.softmax(pred_victim, dim=1)

    else:
        raise ValueError(args.loss)


    loss_values = -loss_fn(pred_clone, pred_victim, reduction='mean')
    # print("True mean loss", loss_values)
    loss_values.backward()

    clone_model.train()
    
    return x_copy.grad, loss_values

def get_pseudo_label(n_or_label, num_classes, device, onehot=False):
    if isinstance(n_or_label, int):
        label = torch.randint(0, num_classes, size=(n_or_label,), device=device)
    else:
        label = n_or_label.to(device)
    if onehot:
        label = torch.zeros(len(label), num_classes, device=device).scatter_(1, label.unsqueeze(1), 1.)
    return label

@torch.no_grad()
def distributed_sinkhorn(out, args):
    # Sinkhorn method for online clustering.
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

class MemoryBank(object):
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn( max_size, dim_feat ).to(device)
        self._ptr = 0
        self.n_updates = 0

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        n, c = feat.shape
        assert self.dim_feat==c and self.max_size % n==0, "%d, %d"%(dim_feat, c, max_size, n)
        self.data[self._ptr:self._ptr+n] = feat.detach()
        self._ptr = (self._ptr+n) % (self.max_size)
        self.n_updates+=n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size
        assert k <= self.max_size

        if self.n_updates>self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index

def clip_images(image_tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

    
def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            img.save(output_filename+'-%d.png'%(idx))

def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack

def flatten_dict(dic):
    flattned = dict()
    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten( k, v )
                else:
                    _flatten( prefix+'/%s'%k, v )
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[ prefix+'/%s'%k ] = v
        
    _flatten(None, dic)
    return flattned

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

def load_yaml(filepath):
    yaml=YAML()  
    with open(filepath, 'r') as f:
        return yaml.load(f)

def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance( postfix, str):
        postfix = [ postfix ]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith( pos ):
                    images.append( os.path.join( dirpath, f ) )
    return images

class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root) #[ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open( self.images[idx] )
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s'%(self.root, len(self), self.transform)

class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.categories = [ int(f) for f in os.listdir( root ) ]
        images = []
        targets = []
        for c in self.categories:
            category_dir = os.path.join( self.root, str(c))
            _images = [ os.path.join( category_dir, f ) for f in os.listdir(category_dir) ]
            images.extend(_images)
            targets.extend([c for _ in range(len(_images))])
        self.images = images
        self.targets = targets
        self.transform = transform
    def __getitem__(self, idx):
        img, target = Image.open( self.images[idx] ), self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.images)

class FeaturePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.datas = self._init_buffer()
        self._idx = 0

    def _init_buffer(self):
        if os.path.exists(self.root + '/buffer.pt'):
            buffer = torch.load(self.root + '/buffer.pt', map_location='cpu')
            buffer = list(buffer)
        else:
            buffer = []
        return buffer

    def add(self, feat, replace=False):
        # print(feat.shape)
        # print(feat.shape)
        if replace:
            self.datas = list(feat.detach().cpu())
        else:
            self.datas.extend(list(feat.detach().cpu()))
        self._idx+=1

    def save_buffer(self):
        # print(self.datas[0].shape)
        # save_x = torch.cat(self.datas, 0)
        save_x = torch.stack(self.datas, 0)
        # assert len(save_x.shape) == 2
        torch.save(save_x, os.path.join(self.root, 'buffer.pt'))

    def get_dataset(self):
        dst = FeatureMemory(self.datas)
        return dst

class ImagePool(object):
    def __init__(self, root, save=True):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        if not save:
            self.datas = self._init_buffer()
        self._idx = 0
        self.save = save

    def _init_buffer(self):
        if os.path.exists(self.root + '/buffer.npy'):
            buffer = np.load(self.root + '/buffer.npy')
            buffer = [Image.fromarray(x) for x in buffer]
        else:
            buffer = []
        return buffer

    def add(self, imgs, targets=None):
        # print(imgs.shape)
        if self.save:
            save_image_batch(imgs, os.path.join( self.root, "%d.png"%(self._idx) ), pack=False)
        else:
            # print(imgs.detach().cpu().clamp_(0,1).permute(0,2,3,1).numpy()[0].shape)
            x = [Image.fromarray(w)  for w in (imgs.detach().cpu().clamp_(0,1).permute(0,2,3,1).numpy()*255).astype(np.uint8)]
            self.datas.extend(x)
            self.save_buffer()
            # print(len(self.datas))
        # self.datas.append(Image.fromarray(imgs.detach().cpu().permute()))
        self._idx+=1

    def save_buffer(self):
        save_x = [np.asarray(x) for x in self.datas]
        numpy_x = np.stack(save_x, 0)
        np.save( os.path.join(self.root, 'buffer.npy'), numpy_x)

    def get_dataset(self, transform=None):
        if self.save:
            return UnlabeledImageDataset(self.root, transform=transform)
        else:
            dst = UnlabelBufferDataset(self.datas, transform=transform)
        
            return dst

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

class UnlabelBufferDataset(Dataset):
    def __init__(self, data, transform):

        self.buffer = data

        self.transform = transform
        # print(self.transform)

    def __getitem__(self, index):
        this_data = self.buffer[index]
        return self.transform(this_data)

    def __len__(self):
        return len(self.buffer)


class FeatureMemory(Dataset):
    def __init__(self, data):
        self.buffer = data

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

class Queue:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.reset()

    def put(self, data, indice):
        # data.shape = (N_batch, D)
        # indice.shape = (N_an, n_batch_neg)
        # Real Data: data[indice], shape = (N_an, n_batch_neg, D)
        assert len(data.shape) == 2
        if len(self.data) == self.capacity:
            self.data.pop(0)
            self.indice.pop(0)
        # print(data.shape)
        self.data.append(data.detach().cpu())
        self.indice.append(indice.cpu())

    def reset(self):
        self.data = []
        self.indice = []
    
    def get(self):
        return self.data[-1]

    def all_batch_num(self):
        if len(self.data) == 0:
            return 0
        return len(self.data) * self.indice[0].size(1)
    
    def sample(self, n_neg, anchor):
        # anchor.shape = (n_an, D)
        # output.shape = (n_an, n_neg)
        # indice.shape = (n_an, n_neg)
        # N_batch, D = self.data[0].size()
        # all_data = torch.cat(self.data, 0)
        # neg_idx = np.random.choice(all_data.size(0), n_neg, replace=False)
        # # neg_data = all_data[neg_idx]
        # interval, samples = neg_idx // N_batch, neg_idx % N_batch
        all_size = len(self.data) * self.indice[0].size(1)
        # outputs = []
        # for i, (data, idx) in enumerate(zip(self.data, self.indice)):
        #     real_data = data[idx]
        #     outputs.append(real_data)
        # real_datas = torch.cat(outputs, 1)
        neg_idx = np.random.choice(all_size, n_neg, replace=False)
        real_datas = torch.cat([data[idx] for data, idx in zip(self.data, self.indice)], 1)
        anchor = F.normalize(anchor, dim=-1)
        real_datas = F.normalize(real_datas, dim=-1)

        return torch.bmm(anchor.unsqueeze(1), real_datas[:, neg_idx, :].permute(0, 2, 1)).squeeze()


@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass

def copy_state_dict(G1, G2, l):
    '''
    G1: l - 1 th generator.
    G2: l th generator.
    copy l th generator's lower parameters to (l-1)-th generator.
    '''
    G1.project.load_state_dict(G2.project.state_dict())
    G1.main[:-(4*l-2)].load_state_dict(G2.main[:-(4*l-2)].state_dict())
    G1.trans_convs[-l].load_state_dict(G2.trans_convs[-l].state_dict())

def get_alpha_adv(epoch, args, ori_adv, type='constant'):
    if epoch  > int(args.epochs * args.begin_fraction) and epoch < int(args.epochs * args.end_fraction) and args.curr_option != 'none': 
        if type == 'linear':
            return ori_adv + (epoch - int(args.epochs * args.begin_fraction)) * args.grad_adv
        elif type == 'interval':
            return ori_adv + args.grad_adv
        else:
            return ori_adv
    elif epoch >= int(args.epochs * args.end_fraction):
        if type == 'linear':
            return ori_adv + (int(args.epochs * args.end_fraction) - int(args.epochs * args.begin_fraction)) * args.grad_adv
        elif type == 'interval':
            return ori_adv + 2 * args.grad_adv
        else:
            return ori_adv
    else:
        return ori_adv

def difficulty_mining(t_feat, s_feat, hard_factor=0., tau=0.07, device='cpu'):
    # print(t_feat.shape, s_feat.shape)
    if t_feat.size()[-1] != s_feat.size()[-1]:
        project_layer = torch.nn.Linear(s_feat.size()[-1], t_feat.size()[-1]).to(device)
        # print(project_layer)
        s_feat = project_layer(s_feat)
    normalized_t, normalized_s = F.normalize(t_feat, dim=-1), F.normalize(s_feat, dim=-1)
    
    d = torch.mm(normalized_t, normalized_s.T)
    n = d.size(0)
    # Div(t, s) at feature map space.
    p = torch.softmax(d / tau, 1)
    p_s_t = torch.softmax(d/tau, 0)
    # print(p, d)
    # exit(-1)
    loss_s_t = -p_s_t.log().diag().mean()
    # InfoNCE base, instance level supervision
    # Maximize MI between teacher and student
    label = torch.arange(len(t_feat), dtype=torch.long, device=device)
    loss_infonce = F.cross_entropy(d / tau, label, reduction='mean')
    return loss_s_t, loss_infonce



class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    Here use framework of MoCo, does not including mementum updating and encoding.
    """
    def __init__(self, dim=128, K=65536, T=0.07, device='cpu', distributed=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999, not implemented here.)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        # self.m = m
        self.T = T
        self.distributed = distributed

        # create the encoders
        # Not implemented in AdaDFKD, because the encoder is the teacher and student mode themselves.

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.device = device

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.distributed:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, hard_factor=0., length=0.5):
        """
        Input:
            im_q: a batch of query feats, usually teacher feature.
            im_k: a batch of key feats, usually student feature
        Output:
            logits, targets
        """

        # compute query features
        # q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(im_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.distributed:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(im_k, dim=1)

            # undo shuffles
            if self.distributed:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach().to(self.device)])

        # Difficulty curriculum adjusting, Motivated by MocoRING
        l_neg_sorted, l_indices = torch.sort(l_neg, 1)
        # From easiest to hardest
        n, ks = l_neg.size() 
        ring_indices = l_indices[:, int(hard_factor * ks) : int((hard_factor + length) * ks)]
        l_neg = torch.gather(l_neg, dim=1, index=ring_indices)

        # logits: Nx(1+K*length)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        # self._dequeue_and_enqueue(q)

        loss = F.cross_entropy(logits, labels, reduction='mean')

        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output