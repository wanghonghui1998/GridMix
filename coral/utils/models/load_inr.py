from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn 

from coral.fourier_features import ModulatedFourierFeatures
from coral.mfn import FourierNet, HyperMultiscaleBACON
from coral.siren import ModulatedSiren, ModulatedSSN, ModulatedSirenGrids, ModulatedSirenOneGrids

NAME_TO_CLASS = {
    "siren": ModulatedSiren,
    "mfn": FourierNet,
    "bacon": HyperMultiscaleBACON,
    "fourier_features": ModulatedFourierFeatures,
}


def create_inr_instance(cfg, input_dim=1, output_dim=1, device="cuda"):
    device = torch.device(device)
    if cfg.inr.model_type == "siren":
        # print(cfg.inr.siren_init)
        inr = ModulatedSiren(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            w0=cfg.inr.w0,
            w0_initial=cfg.inr.w0,
            use_bias=True,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            use_latent=cfg.inr.use_latent,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
            last_activation=cfg.inr.last_activation,
        ).to(device)
    
    elif cfg.inr.model_type == "siren_grid":
        print(cfg.inr.siren_init)
        inr = ModulatedSirenGrids(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            w0=cfg.inr.w0,
            w0_initial=cfg.inr.w0,
            use_bias=True,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            use_latent=cfg.inr.use_latent,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
            last_activation=cfg.inr.last_activation,
            use_norm=cfg.inr.use_norm,
            grid_size=cfg.inr.grid_size,
            siren_init=cfg.inr.siren_init,
        ).to(device)
    
    elif cfg.inr.model_type == "siren_one_grid":
        print(cfg.inr.siren_init)
        inr = ModulatedSirenOneGrids(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            w0=cfg.inr.w0,
            w0_initial=cfg.inr.w0,
            use_bias=True,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            use_latent=cfg.inr.use_latent,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
            last_activation=cfg.inr.last_activation,
            use_norm=cfg.inr.use_norm,
            grid_size=cfg.inr.grid_size,
            siren_init=cfg.inr.siren_init,
        ).to(device)

    elif cfg.inr.model_type == "ssn":
        inr = ModulatedSSN(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            w0=cfg.inr.w0,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
        ).to(device)
    elif cfg.inr.model_type == "mfn":
        inr = FourierNet(
            input_dim,
            cfg.inr.hidden_dim,
            cfg.inr.latent_dim,
            output_dim,
            cfg.inr.depth,
            input_scale=cfg.inr.input_scale,
        ).to(device)

    elif cfg.inr.model_type == "bacon":
        mod_activation = (
            cfg.inr.mod_activation if cfg.inr.mod_activation != "None" else None
        )
        try:
            # if input_scales look like '0.125', '0.125', etc.
            input_scales = [float(v) for v in cfg.inr.input_scales]
        except ValueError:
            # if input_scales look like '1./8', '1./8', etc.
            input_scales = [
                float(v.split("/")[0]) / float(v.split("/")[1]) for v in input_scales
            ]
        inr = HyperMultiscaleBACON(
            input_dim,
            cfg.inr.hidden_dim,
            output_dim,
            hidden_layers=len(input_scales) - 1,
            bias=True,
            frequency=cfg.inr.frequency,
            quantization_interval=cfg.inr.quantization_multiplier * np.pi,
            input_scales=input_scales,
            output_layers=cfg.inr.output_layers,
            reuse_filters=False,
            use_latent=True,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
            filter_type=cfg.inr.filter_type,
            mod_activation=mod_activation,
        ).to(device)

    elif cfg.inr.model_type == "fourier_features":
        inr = ModulatedFourierFeatures(
            input_dim=input_dim,
            output_dim=output_dim,
            num_frequencies=cfg.inr.num_frequencies,
            latent_dim=cfg.inr.latent_dim,
            width=cfg.inr.hidden_dim,
            depth=cfg.inr.depth,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            frequency_embedding=cfg.inr.frequency_embedding,
            include_input=cfg.inr.include_input,
            scale=cfg.inr.scale,
            max_frequencies=cfg.inr.max_frequencies,
            base_frequency=cfg.inr.base_frequency,
        ).to(device)

    else:
        raise NotImplementedError(f"No corresponding class for {cfg.inr.model_type}")

    return inr


def load_inr_model(
    run_dir, run_name, data_to_encode, input_dim=1, output_dim=1, device="cuda"
):  
    inr_train = torch.load(run_dir / f"{run_name}.pt")

    inr_state_dict = inr_train["inr"]
    cfg = inr_train["cfg"]
    alpha = inr_train["alpha"]
    print(f'{run_name}, epoch {inr_train["epoch"]}, alpha1 {alpha.item()}')
    inr = create_inr_instance(cfg, input_dim, output_dim, device)
    inr.load_state_dict(inr_state_dict)
    # new_state_dict = {}
    # for name, para in inr_state_dict.items():
    #     if 'alpha' in name:
    #         print(f'alpha2 {para}')
    #         pass 
    #     else:
    #         new_state_dict[name[7:]] = para 
    # inr.load_state_dict(new_state_dict)
    inr.eval()

    return inr, alpha

def load_inr_model_dino(
    run_dir, run_name, data_to_encode, input_dim=1, output_dim=1, device="cuda"
):  
    inr_train = torch.load(run_dir / f"{run_name}.pt")

    inr_state_dict = inr_train["inr"]
    cfg = inr_train["cfg"]
    # alpha = inr_train["alpha"]
    # print(f'alpha1 {alpha}')
    inr = create_inr_instance(cfg, input_dim, output_dim, device)
    # inr.load_state_dict(inr_state_dict)
    new_state_dict = {}
    modulations = inr_state_dict['code'].detach()
    for name, para in inr_state_dict.items():
        if name == 'code':
            print(f'code')
            pass 
        else:
            new_state_dict[name[4:]] = para # remove 'inr.' 
    inr.load_state_dict(new_state_dict)
    inr.eval()

    return inr, modulations

def load_inr_model_edg(
    run_dir, run_name, data_to_encode, input_dim=1, output_dim=1, device="cuda"
):  
    inr_train = torch.load(run_dir / f"{run_name}.pt")

    inr_state_dict = inr_train["inr"]
    cfg = inr_train["cfg"]
    # alpha = inr_train["alpha"]
    # print(f'alpha1 {alpha}')
    inr = create_inr_instance(cfg, input_dim, output_dim, device)
    # inr.load_state_dict(inr_state_dict)
    new_state_dict = {}
    code = inr_state_dict['code'].detach()
    alpha = inr_state_dict['alpha'].detach()
    print(f'alpha {alpha}')
    for name, para in inr_state_dict.items():
        if name == 'code' or name == 'alpha' or 'cls' in name:
            print(f'code or alpha or cls')
            pass 
        else:
            new_state_dict[name[4:]] = para # remove 'inr.' 
    inr.load_state_dict(new_state_dict)
    inr.eval()

    return inr, alpha, code


def load_inr_model_enc(
    run_dir, run_name, data_to_encode, input_dim=1, output_dim=1, device="cuda"
):  
    inr_train = torch.load(run_dir / f"{run_name}.pt")

    inr_state_dict = inr_train["inr"]
    cfg = inr_train["cfg"]
    # alpha = inr_train["alpha"]
    # print(f'alpha1 {alpha}')
    inr = create_inr_instance(cfg, input_dim, output_dim, device)
    code = FNO2d(12, 12, 32, cfg.inr.latent_dim).to(device)
    # inr.load_state_dict(inr_state_dict)
    new_inr_state_dict = {}
    new_code_state_dict = {}
    # modulations = inr_state_dict['code'].detach()
    for name, para in inr_state_dict.items():
        if 'code' in name:
            print(f'code')
            new_code_state_dict[name[5:]] = para
        else:
            print(f'inr')
            new_inr_state_dict[name[4:]] = para # remove 'inr.' 
    inr.load_state_dict(new_inr_state_dict)
    inr.eval()
    code.load_state_dict(new_code_state_dict)
    code.eval()
    return inr, code

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_channels):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        # self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2 + 1, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, num_channels)
        # self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        # x = x.permute(0, 2, 3, 1)
        x = x.mean(-1).mean(-1)
        x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)