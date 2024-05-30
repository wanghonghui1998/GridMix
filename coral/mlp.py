from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np 

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


ACTIVATIONS = {
    "relu": partial(nn.ReLU),
    "sigmoid": partial(nn.Sigmoid),
    "tanh": partial(nn.Tanh),
    "selu": partial(nn.SELU),
    "softplus": partial(nn.Softplus),
    "gelu": partial(nn.GELU),
    "swish": partial(Swish),
    "elu": partial(nn.ELU),
    "leakyrelu": partial(nn.LeakyReLU),
}


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        drop_rate=0.0,
        activation="swish",
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.activation1 = ACTIVATIONS[activation]()
        self.activation2 = ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        eta = self.linear1(x)
        # eta = self.batch_norm1(eta)
        eta = self.linear2(self.activation1(eta))
        # no more dropout
        # out = self.activation2(x + self.dropout(eta))
        out = x + self.activation2(self.dropout(eta))
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        depth=2,
        dropout=0.0,
        activation="swish",
    ):
        super().__init__()
        net = [ResBlock(input_dim, hidden_dim, dropout, activation)]
        for _ in range(depth - 1):
            net.append(ResBlock(input_dim, hidden_dim, dropout, activation))

        self.net = nn.Sequential(*net)
        self.project_map = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        out = self.net(z)
        out = self.project_map(out)

        return out


class ResNetDynamics(nn.Module):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        depth=2,
        dropout=0.0,
        activation="swish",
        dt= 0.5/250
    ):
        super().__init__()
        net = [ResBlock(input_dim, hidden_dim, dropout, activation)]
        for _ in range(depth - 1):
            net.append(ResBlock(input_dim, hidden_dim, dropout, activation))

        self.net = nn.Sequential(*net)
        self.project_map = nn.Linear(input_dim, output_dim)
        self.dt = dt

    def forward(self, z):
        # input (b, l, t)
        T = z.shape[-1]
        z_last = z[..., -1].unsqueeze(-1)
        z = einops.rearrange(z, 'b l t -> b (l t)')
        out = self.net(z)
        out = self.project_map(out)

        out = einops.rearrange(out, 'b (l t) -> b l t', t=T)

        #dt = (torch.ones(1, 1, T) * self.dt).to(out.device)
        #dt = torch.cumsum(dt, dim=2)

        out = z_last + out
        return out  


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        depth=2,
        dropout=0.0,
        activation="swish",
    ):
        super().__init__()
        self.activation = ACTIVATIONS[activation]
        net = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(depth - 1):
            net.append(self.activation())
            net.append(nn.Dropout(dropout))
            net.append(nn.Linear(hidden_dim, hidden_dim))
        net.append(self.activation())

        self.net = nn.Sequential(*net)
        self.dropout = nn.Dropout(dropout)
        self.project_map = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = self.net(z)
        # z = self.dropout(z)
        out = self.project_map(z)

        return out


class MLP2(nn.Module):
    def __init__(self, code_size, hidden_size, depth=1, nl="swish"):
        super().__init__()

        net = [nn.Linear(code_size, hidden_size), ACTIVATIONS[nl]()]

        for j in range(depth - 1):
            net.append(nn.Linear(hidden_size, hidden_size))
            net.append(ACTIVATIONS[nl]())

        net.append(nn.Linear(hidden_size, code_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class MLP3(nn.Module):
    def __init__(self, code_size, memory_size, hidden_size, depth=1, nl="swish"):
        super().__init__()

        net = [nn.Linear(code_size+memory_size, hidden_size), ACTIVATIONS[nl]()]

        for j in range(depth - 1):
            net.append(nn.Linear(hidden_size, hidden_size))
            net.append(ACTIVATIONS[nl]())

        net.append(nn.Linear(hidden_size, code_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class SetEncoder(nn.Module):
    def __init__(self, code_size, n_cond, hidden_size, out_size=None, nl='swish'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            ACTIVATIONS[nl](),
            nn.Linear(hidden_size, hidden_size),
            ACTIVATIONS[nl](),
            nn.Linear(hidden_size, hidden_size),
            ACTIVATIONS[nl](),
            nn.Linear(hidden_size, code_size if out_size == None else out_size),
        )
        self.ave = nn.Conv1d(code_size, code_size, n_cond)

    def forward(self, x):
        ''' x: b,t,c '''
        aggreg = self.net(x)
        return self.ave(aggreg.permute(0, 2, 1)).permute(0, 2, 1).squeeze()

class Derivative(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, depth=2, **kwargs):
        super().__init__()
        input_dim = code_c * state_c
        self.net = MLP2(input_dim, hidden_c, depth=depth, nl="swish")

    def forward(self, t, u):
        return self.net(u)

class Derivative2nd(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, depth=2, **kwargs):
        super().__init__()
        input_dim = code_c * state_c
        
        self.net = MLP3(input_dim, input_dim, hidden_c, depth=depth, nl="swish")
    
    def forward(self, t, u_c):
        u,c = u_c
   
        du_dt = self.net(torch.cat([u, c], dim=-1))
        dc_dt = u 
        # print(f'u {u.abs().max()}, c {c.abs().max()}, m {m.abs().max()}, dc_dt {dc_dt.abs().max()}')
        return (du_dt, dc_dt)

class DerivativeHippo(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, depth=2, measure='legs', N=16, memory_size=0, **kwargs):
        super().__init__()
        input_dim = code_c * state_c
        
        self.memory_size = memory_size
        if self.memory_size > 0:
            self.code_to_memory = nn.Linear(input_dim, memory_size)
            self.net = MLP3(input_dim, memory_size*N, hidden_c, depth=depth, nl="swish")
        else:
            self.net = MLP3(input_dim, input_dim*N, hidden_c, depth=depth, nl="swish")
        A,B = transition(measure, N)
        B = B.squeeze(-1)
        self.register_buffer('A', torch.tensor(A).float()) # (N, N)
        self.register_buffer('B', torch.tensor(B).float()) # (N,)

    def forward(self, t, u_c):
        u,c = u_c
        # import pdb; pdb.set_trace()
        shape0,shape1,shape2 = c.shape  
        # u: B,D; c: B,D,N
        if self.memory_size > 0:
            m = self.code_to_memory(u)
        else:
            m = u 
        dc_dt = 1.0 / t * torch.matmul(self.A, c.unsqueeze(-1)).squeeze(-1) + 1.0 / t * self.B * m.unsqueeze(-1)
        c = c.reshape(shape0, shape1*shape2)
        du_dt = self.net(torch.cat([u, c], dim=-1))
        # print(f'u {u.abs().max()}, c {c.abs().max()}, m {m.abs().max()}, dc_dt {dc_dt.abs().max()}')
        return (du_dt, dc_dt)


def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures.

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    
    if measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]

    return A, B





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
        self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2 + num_channels, self.width)
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

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)
        # self.grid = None 
        self.grid_per_bs = None

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        # print(x.shape, grid.shape)
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
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        if self.grid_per_bs is not None:
            return self.grid_per_bs.repeat(shape[0],1,1,1)
        else:
            batchsize, size_x, size_y = shape[0], shape[1], shape[2]
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat(
                [batchsize, 1, size_y, 1])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat(
                [batchsize, size_x, 1, 1])
            grid = torch.cat((gridx, gridy), dim=-1).to(device)
            self.grid_per_bs = grid[0:1]
            return grid
        # return torch.cat((gridx, gridy), dim=-1).to(device)

class DerivativeFNO(nn.Module):
    def __init__(self, modes1, modes2, width, grid_channel=6, grid_size=64, **kwargs):
        super().__init__()
        self.grid_channel = grid_channel
        self.grid_size = grid_size
        self.net = FNO2d(modes1, modes2, width, grid_channel)

    def forward(self, t, u):
        u_grid = u.reshape(u.shape[0], self.grid_channel, self.grid_size, self.grid_size).permute(0,2,3,1)
        grad_grid = self.net(u_grid)
        return grad_grid.permute(0,3,1,2).reshape(u.shape[0], -1)