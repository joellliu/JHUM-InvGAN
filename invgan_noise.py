from art.defences.preprocessor.preprocessor import Preprocessor
from styleGAN import Generator
from resisc_encoder_resunet_noise import Encoder
from losses import PerceptualLoss
from torch import optim
from advertorch.utils import NormalizeByChannelMeanStd
from tensorflow.keras.preprocessing import image
import numpy as np
import torch
import math
import pdb
from tqdm import tqdm

def make_noise(batch_size, log_size, device='cuda'):
    noises = [torch.randn(batch_size, 1, 2 ** 2, 2 ** 2, device=device)]

    for i in range(3, log_size + 1):
        for _ in range(2):
            noises.append(torch.randn(batch_size, 1, 2 ** i, 2 ** i, device=device))

    return noises

def noise_regularize(noises, batch_size):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = loss + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) \
                   + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)

            if size <= 8:
                break

            noise = noise.reshape([batch_size, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

class InvGAN(Preprocessor):
    """
        Unnormalize inputs that were normalized during preprocessing,
        project onto manifold, and renormalize
    """
    def __init__(
        self,
        clip_values,
        step=200,
        means=None,
        stds=None,
        gan_ckpt='styleGAN.pt',
        encoder_ckpt='encoder.pt',
        optimize_noise=True,
        use_noise_regularize=False,
        use_lpips=False,
        apply_fit=False,
        apply_predict=True,
        mse=500,
        lr_rampup=0.05,
        lr_rampdown=0.05,
        noise=0.05,
        noise_ramp=0.75,
        noise_regularize=1e5,
        lr=0.1
    ):
        
        super(InvGAN, self).__init__()
        #print("invgan")
        #pdb.set_trace()
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        # setup normalization parameters
        if means is None:
            means = (0.0, 0.0, 0.0)  # identity operation
        if len(means) != 3:
            raise ValueError("means must have 3 values, one per channel")
        self.means = means

        if stds is None:
            stds = (1.0, 1.0, 1.0)  # identity operation
        if len(stds) != 3:
            raise ValueError("stds must have 3 values, one per channel")
        self.stds = stds
        self.clip_values = clip_values

        # setup optimization parameters
        self.optimize_noise = optimize_noise
        self.use_noise_regularize = use_noise_regularize
        self.use_lpips = use_lpips
        self.step = step
        self.mse = mse
        self.lr = lr
        self.lr_rampup = lr_rampup
        self.lr_rampdown = lr_rampdown
        self.noise = noise
        self.noise_ramp = noise_ramp
        self.noise_regularize = noise_regularize

        # setup generator
        self.generator = Generator(256, 512, 8)
        self.generator.load_state_dict(torch.load(gan_ckpt)['g_ema'])
        self.generator.eval()
        self.generator.cuda()
        self.deprocess_layer = NormalizeByChannelMeanStd([-1., -1., -1.], [2., 2., 2.]).cuda()

        # setup encoder
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(encoder_ckpt)['netE'])
        self.encoder.eval()
        self.encoder.cuda()

        # setup loss
        if use_lpips:
            self.lpips = PerceptualLoss().cuda()

        # estimate latent code statistics
        n_mean_latent = 10000
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device='cuda')
            latent_out = self.generator.style(noise_sample)

            latent_mean = latent_out.mean(0)
            self.latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    def __call__(self, x, y=None):
        device = 'cuda'
        batch_size = x.shape[0]
        # denormalize images to [0, 1]
        x = x * self.stds + self.means
        np.clip(x, self.clip_values[0], self.clip_values[1], x)
        # normalize images to [-1, 1]
        x = 2*x - 1
        #pdb.set_trace()
        x = torch.from_numpy(x)
        # convert from NHWC to NCHW
        x = x.permute(0, 3, 1, 2).type('torch.FloatTensor')
        x = x.to(device)
        # reshape if input shape is not (256, 256)
        if x.shape[2] != 256:
            x = torch.nn.Upsample((256, 256), mode='bicubic')(x)
        # encode images
        latent_in, noises = self.encoder(x)
        latent_in = latent_in.detach().clone()
        latent_in.cuda()
        latent_in.requires_grad = True
        # optimize latent code
        tmps = []
        for noise in noises:
            tmp = noise.detach().clone()
            tmp.requires_grad = self.optimize_noise
            tmps.append(tmp)
        noises = tmps
        if self.optimize_noise:
            optimizer = optim.Adam([latent_in] + noises, lr=self.lr)
        else:
            optimizer = optim.Adam([latent_in], lr=self.lr)
        
        pbar = tqdm(range(self.step))
        
        for i in pbar:
            t = i/self.step
            lr = get_lr(t, self.lr, rampdown=0.1)
            optimizer.param_groups[0]['lr'] = lr
            noise_strength = self.latent_std * self.noise * max(0, 1 - t / self.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            img_gen, _ = self.generator([latent_n], input_is_latent=True, noise=noises)
            # compute loss
            mse_loss = torch.nn.functional.mse_loss(img_gen, x)
            loss = self.mse * mse_loss
            if self.use_lpips:
                p_loss = self.lpips(self.deprocess_layer(x), self.deprocess_layer(img_gen)).sum() / batch_size
                loss += p_loss
            if self.use_noise_regularize & self.optimize_noise:
                n_loss = self.noise_regularize * noise_regularize(noises, batch_size)
                loss += n_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            noise_normalize_(noises)
            description = f'mse: {mse_loss.item():.4f}; '
            if self.use_lpips:
                description += f'perceptual: {p_loss.item():.4f}; '
            if self.use_noise_regularize & self.optimize_noise:
                description += f'n_loss: {n_loss.item():.4f}; '
            pbar.set_description((description))
        # project images
        x, _ = self.generator([latent_in], input_is_latent=True, noise=noises)
        # reshape to (224, 224)
        x = torch.nn.Upsample((224, 224), mode='bicubic')(x)
        x = x.detach().clamp_(min=-1, max=1).add(1).div_(2).permute(0, 2, 3, 1).to('cpu').numpy()  # x in [0, 1]
        # renormalize images
        x = (x - self.means) / self.stds
        return x, y
    
    @property
    def apply_fit(self) -> bool:
        return self._apply_fit
    @property
    def apply_predict(self) -> bool:
        return self._apply_predict
    
    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
    pass

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        #pdb.set_trace()
        return grad