# Based on https://github.com/EmilienDupont/coin
from math import sqrt

import einops
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# import tinycudann as tcnn 
#from coral.utils.interpolate import knn_interpolate_custom, rescale_coordinate

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model. If it is, no
            activation is applied and 0.5 is added to the output. Since we
            assume all training data lies in [0, 1], this allows for centering
            the output of the model.
        use_bias (bool): Whether to learn bias in linear layer.
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
        siren_init=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        # print(siren_init)
        # import pdb; pdb.set_trace()
        print(f'w0 {w0}')
        if siren_init:
            print(f'siren init')    
            w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
            nn.init.uniform_(self.linear.weight, -w_std, w_std)
            if use_bias:
                nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if self.is_last:
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            out += 0
        else:
            out = self.activation(out)
        return out

class Siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        siren_init=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers

        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    siren_init=siren_init
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayer(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias, is_last=True, siren_init=siren_init
        )

    def forward(self, x):
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        x = self.net(x)
        return self.last_layer(x)

class ModulatedSirenLR(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.rank = latent_dim // (num_layers-2) // dim_hidden // 2
        self.num_modulations = num_layers - 2
        self.A_bias = 1.0 / dim_hidden 
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

        x = x.view(x.shape[0], -1, x.shape[-1])
        # x: torch.Size([128, 4096, 2])
        # latent: torch.Size([128, 128])
        # modulations: torch.Size([128, 384])
        # Shape (batch_size, num_modulations)
        modulations = latent.reshape(latent.shape[0], self.num_modulations, 2, -1, self.rank)
        # print(modulations.shape)
        # modulations = self.modulation_net(latent)
        
        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        
        for layer_idx, module in enumerate(self.net):
            y = module.linear(x) 
            if layer_idx != 0:
                
                A = modulations[:, layer_idx-1, 0:1]    # (batch_size, 1, dim_hidden, rank)
                B = modulations[:, layer_idx-1, 1:2]    # (batch_size, 1, dim_hidden, rank)
                delta_y = torch.sum(x.unsqueeze(-1) * (A+self.A_bias), dim=-2, keepdim=True)    # (batch_size, num_points, 1, rank)
                # print(f'{layer_idx} after A', delta_y.abs().mean())
                delta_y = torch.sum(delta_y * B, dim=-1)
                # print(f'{layer_idx} after B', delta_y.abs().mean())
                y = y + delta_y / self.rank 

            x = module.activation(y)  # (batch_size, num_points, dim_hidden)

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])
    
class ModulatedSiren(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

        x = x.view(x.shape[0], -1, x.shape[-1])
        # x: torch.Size([128, 4096, 2])
        # latent: torch.Size([128, 128])
        # modulations: torch.Size([128, 384])
        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)
        
        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx: idx +
                                    self.dim_hidden].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, mid_idx + idx: mid_idx + idx + self.dim_hidden
                ].unsqueeze(1)
            else:
                shift = 0.0

            x = module.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenMultiInit(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        siren_init=True,
        num_init=1,
        scale_init=1e-3,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )

        self.latent_init = nn.Parameter(scale_init * torch.randn(num_init, latent_dim))

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

        x = x.view(x.shape[0], -1, x.shape[-1])
        # x: torch.Size([128, 4096, 2])
        # latent: torch.Size([128, 128])
        # modulations: torch.Size([128, 384])
        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)
        
        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx: idx +
                                    self.dim_hidden].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, mid_idx + idx: mid_idx + idx + self.dim_hidden
                ].unsqueeze(1)
            else:
                shift = 0.0

            x = module.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

# class ModulatedSirenHashEnc(Siren):
#     """Modulated SIREN model.

#     Args:
#         dim_in (int): Dimension of input.
#         dim_hidden (int): Dimension of hidden layers.
#         dim_out (int): Dimension of output.
#         num_layers (int): Number of layers.
#         w0 (float): Omega 0 from SIREN paper.
#         w0_initial (float): Omega 0 for first layer.
#         use_bias (bool): Whether to learn bias in linear layer.
#         modulate_scale (bool): Whether to modulate with scales.
#         modulate_shift (bool): Whether to modulate with shifts.
#         use_latent (bool): If true, use a latent vector which is mapped to
#             modulations, otherwise use modulations directly.
#         latent_dim (int): Dimension of latent vector.
#         modulation_net_dim_hidden (int): Number of hidden dimensions of
#             modulation network.
#         modulation_net_num_layers (int): Number of layers in modulation network.
#             If this is set to 1 will correspond to a linear layer.
#     """

#     def __init__(
#         self,
#         dim_in,
#         dim_hidden,
#         dim_out,
#         num_layers,
#         w0=30.0,
#         w0_initial=30.0,
#         use_bias=True,
#         modulate_scale=False,
#         modulate_shift=True,
#         use_latent=False,
#         latent_dim=64,
#         modulation_net_dim_hidden=64,
#         modulation_net_num_layers=1,
#         mu=0,
#         sigma=1,
#         last_activation=None,
#         siren_init=True,
#         n_levels=3,
#         n_features_per_level=2,
#         log2_hashmap_size=5,
#         base_resolution=8,
#         per_level_scale=1.5,
#     ):
#         super().__init__(
#             dim_in+n_levels*n_features_per_level,
#             dim_hidden,
#             dim_out,
#             num_layers,
#             w0,
#             w0_initial,
#             use_bias,
#             siren_init,
#         )
#         # Must modulate at least one of scale and shift
#         assert modulate_scale or modulate_shift
#         self.grid_encoder = tcnn.Encoding(
#             n_input_dims=dim_in,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": n_levels,
#                 "n_features_per_level": n_features_per_level,
#                 "log2_hashmap_size": log2_hashmap_size,
#                 "base_resolution": base_resolution,
#                 "per_level_scale": per_level_scale,
#             },
#         )

#         self.modulate_scale = modulate_scale
#         self.modulate_shift = modulate_shift
#         self.w0 = w0
#         self.w0_initial = w0_initial
#         self.mu = mu
#         self.sigma = sigma
#         self.last_activation = (
#             nn.Identity() if last_activation is None else last_activation
#         )

#         # We modulate features at every *hidden* layer of the base network and
#         # therefore have dim_hidden * (num_layers - 1) modulations, since the
#         # last layer is not modulated
#         num_modulations = dim_hidden * (num_layers - 1)
#         if self.modulate_scale and self.modulate_shift:
#             # If we modulate both scale and shift, we have twice the number of
#             # modulations at every layer and feature
#             num_modulations *= 2

#         if use_latent:
#             self.modulation_net = LatentToModulation(
#                 latent_dim,
#                 num_modulations,
#                 modulation_net_dim_hidden,
#                 modulation_net_num_layers,
#             )
#         else:
#             self.modulation_net = Bias(num_modulations)

#         # Initialize scales to 1 and shifts to 0 (i.e. the identity)
#         if not use_latent:
#             if self.modulate_shift and self.modulate_scale:
#                 self.modulation_net.bias.data = torch.cat(
#                     (
#                         torch.ones(num_modulations // 2),
#                         torch.zeros(num_modulations // 2),
#                     ),
#                     dim=0,
#                 )
#             elif self.modulate_scale:
#                 self.modulation_net.bias.data = torch.ones(num_modulations)
#             else:
#                 self.modulation_net.bias.data = torch.zeros(num_modulations)

#         self.num_modulations = num_modulations

#     def modulated_forward(self, x, latent):
#         """Forward pass of modulated SIREN model.

#         Args:
#             x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
#                 any spatial dimensions, e.g. (height, width), (height * width,)
#                 or (depth, height, width) etc.
#             latent (torch.Tensor): Shape (batch_size, latent_dim). If
#                 use_latent=False, then latent_dim = num_modulations.

#         Returns:
#             Output features of shape (batch_size, *, dim_out).
#         """
#         # Extract batch_size and spatial dims of x, so we can reshape output
#         x_shape = x.shape[:-1]
#         # Flatten all spatial dimensions, i.e. shape
#         # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

#         x = x.view(x.shape[0], -1, x.shape[-1])
#         B_p,N_p,d_p = x.shape
#         x_grid = self.grid_encoder(x.reshape(-1, d_p))
#         x = torch.cat([x, x_grid.reshape(B_p, N_p, -1)], dim=-1)
#         # x: torch.Size([128, 4096, 2])
#         # latent: torch.Size([128, 128])
#         # modulations: torch.Size([128, 384])
#         # Shape (batch_size, num_modulations)
#         modulations = self.modulation_net(latent)
        
#         # Split modulations into shifts and scales and apply them to hidden
#         # features.
#         mid_idx = (
#             self.num_modulations // 2
#             if (self.modulate_scale and self.modulate_shift)
#             else 0
#         )
#         idx = 0
#         for module in self.net:
#             if self.modulate_scale:
#                 # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
#                 # modulations remain zero centered
#                 scale = modulations[:, idx: idx +
#                                     self.dim_hidden].unsqueeze(1) + 1.0
#             else:
#                 scale = 1.0

#             if self.modulate_shift:
#                 # Shape (batch_size, 1, dim_hidden)
#                 shift = modulations[
#                     :, mid_idx + idx: mid_idx + idx + self.dim_hidden
#                 ].unsqueeze(1)
#             else:
#                 shift = 0.0

#             x = module.linear(x)
#             x = scale * x + shift  # Broadcast scale and shift across num_points
#             x = module.activation(x)  # (batch_size, num_points, dim_hidden)

#             idx = idx + self.dim_hidden

#         # Shape (batch_size, num_points, dim_out)
#         out = self.last_activation(self.last_layer(x))
#         out = out * self.sigma + self.mu
#         # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
#         return out.view(*x_shape, out.shape[-1])

# class ModulatedNFFB(Siren):
#     """Modulated SIREN model.

#     Args:
#         dim_in (int): Dimension of input.
#         dim_hidden (int): Dimension of hidden layers.
#         dim_out (int): Dimension of output.
#         num_layers (int): Number of layers.
#         w0 (float): Omega 0 from SIREN paper.
#         w0_initial (float): Omega 0 for first layer.
#         use_bias (bool): Whether to learn bias in linear layer.
#         modulate_scale (bool): Whether to modulate with scales.
#         modulate_shift (bool): Whether to modulate with shifts.
#         use_latent (bool): If true, use a latent vector which is mapped to
#             modulations, otherwise use modulations directly.
#         latent_dim (int): Dimension of latent vector.
#         modulation_net_dim_hidden (int): Number of hidden dimensions of
#             modulation network.
#         modulation_net_num_layers (int): Number of layers in modulation network.
#             If this is set to 1 will correspond to a linear layer.
#     """

#     def __init__(
#         self,
#         dim_in,
#         dim_hidden,
#         dim_out,
#         num_layers,
#         w0=30.0,
#         w0_initial=30.0,
#         use_bias=True,
#         modulate_scale=False,
#         modulate_shift=True,
#         use_latent=False,
#         latent_dim=64,
#         modulation_net_dim_hidden=64,
#         modulation_net_num_layers=1,
#         mu=0,
#         sigma=1,
#         last_activation=None,
#         siren_init=True,
#         n_levels=3,
#         n_features_per_level=2,
#         log2_hashmap_size=5,
#         base_resolution=8,
#         per_level_scale=1.5,
#         base_sigma=5.0,
#         exp_sigma=2.0,
#     ):
#         super().__init__(
#             dim_in,
#             dim_hidden,
#             dim_out,
#             num_layers,
#             w0,
#             w0_initial,
#             use_bias,
#             siren_init,
#         )
#         # Must modulate at least one of scale and shift
#         assert modulate_scale or modulate_shift
#         assert n_levels == num_layers-1
#         self.n_levels = n_levels
#         self.n_features_per_level = n_features_per_level
#         self.grid_encoder = tcnn.Encoding(
#             n_input_dims=dim_in,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": n_levels,
#                 "n_features_per_level": n_features_per_level,
#                 "log2_hashmap_size": log2_hashmap_size,
#                 "base_resolution": base_resolution,
#                 "per_level_scale": per_level_scale,
#             },
#         )

#         ffn_list = []
#         for i in range(n_levels):
#             ffn = torch.randn((n_features_per_level, dim_hidden), requires_grad=True) * base_sigma * exp_sigma ** i
#             ffn_list.append(ffn)
#         self.ffn = nn.Parameter(torch.stack(ffn_list, dim=0))

#         self.modulate_scale = modulate_scale
#         self.modulate_shift = modulate_shift
#         self.w0 = w0
#         self.w0_initial = w0_initial
#         self.mu = mu
#         self.sigma = sigma
#         self.last_activation = (
#             nn.Identity() if last_activation is None else last_activation
#         )

#         # We modulate features at every *hidden* layer of the base network and
#         # therefore have dim_hidden * (num_layers - 1) modulations, since the
#         # last layer is not modulated
#         num_modulations = dim_hidden * (num_layers - 1)
#         if self.modulate_scale and self.modulate_shift:
#             # If we modulate both scale and shift, we have twice the number of
#             # modulations at every layer and feature
#             num_modulations *= 2

#         if use_latent:
#             self.modulation_net = LatentToModulation(
#                 latent_dim,
#                 num_modulations,
#                 modulation_net_dim_hidden,
#                 modulation_net_num_layers,
#             )
#         else:
#             self.modulation_net = Bias(num_modulations)

#         # Initialize scales to 1 and shifts to 0 (i.e. the identity)
#         if not use_latent:
#             if self.modulate_shift and self.modulate_scale:
#                 self.modulation_net.bias.data = torch.cat(
#                     (
#                         torch.ones(num_modulations // 2),
#                         torch.zeros(num_modulations // 2),
#                     ),
#                     dim=0,
#                 )
#             elif self.modulate_scale:
#                 self.modulation_net.bias.data = torch.ones(num_modulations)
#             else:
#                 self.modulation_net.bias.data = torch.zeros(num_modulations)

#         self.num_modulations = num_modulations

#     def modulated_forward(self, x, latent):
#         """Forward pass of modulated SIREN model.

#         Args:
#             x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
#                 any spatial dimensions, e.g. (height, width), (height * width,)
#                 or (depth, height, width) etc.
#             latent (torch.Tensor): Shape (batch_size, latent_dim). If
#                 use_latent=False, then latent_dim = num_modulations.

#         Returns:
#             Output features of shape (batch_size, *, dim_out).
#         """
#         # Extract batch_size and spatial dims of x, so we can reshape output
#         x_shape = x.shape[:-1]
#         # Flatten all spatial dimensions, i.e. shape
#         # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

#         x = x.view(x.shape[0], -1, x.shape[-1])
#         B_p,N_p,d_p = x.shape
#         x_grid = self.grid_encoder(x.reshape(-1, d_p)).view(B_p,N_p,self.n_levels,self.n_features_per_level).permute(2,0,1,3)
#         embedding_list = []
#         for i in range(self.n_levels):
#             grid_output = torch.matmul(x_grid[i].float(), self.ffn[i])
#             # grid_output = torch.sin(2 * torch.pi * grid_output)
#             embedding_list.append(grid_output)

#         # x = torch.cat([x, x_grid.reshape(B_p, N_p, -1)], dim=-1)
#         # x: torch.Size([128, 4096, 2])
#         # latent: torch.Size([128, 128])
#         # modulations: torch.Size([128, 384])
#         # Shape (batch_size, num_modulations)
#         modulations = self.modulation_net(latent)
        
#         # Split modulations into shifts and scales and apply them to hidden
#         # features.
#         mid_idx = (
#             self.num_modulations // 2
#             if (self.modulate_scale and self.modulate_shift)
#             else 0
#         )
#         idx = 0
#         for l_idx, module in enumerate(self.net):
#             if self.modulate_scale:
#                 # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
#                 # modulations remain zero centered
#                 scale = modulations[:, idx: idx +
#                                     self.dim_hidden].unsqueeze(1) + 1.0
#             else:
#                 scale = 1.0

#             if self.modulate_shift:
#                 # Shape (batch_size, 1, dim_hidden)
#                 shift = modulations[
#                     :, mid_idx + idx: mid_idx + idx + self.dim_hidden
#                 ].unsqueeze(1)
#             else:
#                 shift = 0.0

#             x = module.linear(x)
#             x = scale * x + shift + embedding_list[l_idx] # Broadcast scale and shift across num_points
#             x = module.activation(x)  # (batch_size, num_points, dim_hidden)
#             # x = x + embedding_list[l_idx]
#             idx = idx + self.dim_hidden

#         # Shape (batch_size, num_points, dim_out)
#         out = self.last_activation(self.last_layer(x))
#         out = out * self.sigma + self.mu
#         # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
#         return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsFourier(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        cos_term = torch.cos(angles)
        sin_term = torch.sin(angles)
        self.register_buffer('cos_term', cos_term)
        self.register_buffer('sin_term', sin_term)

        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, 2, 1, 1, self.grid_size, self.grid_size)
        latent_A = latent[:, :, 0]
        latent_B = latent[:, :, 1]
        latent = (latent_A * self.cos_term - latent_B * self.sin_term).reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size, -1).sum(-1)

        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenGridsFourierC(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        cos_term = torch.cos(angles)
        sin_term = torch.sin(angles)
        self.register_buffer('cos_term', cos_term)
        self.register_buffer('sin_term', sin_term)
        self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, 2, 1, self.grid_size, self.grid_size)
        latent_A = latent[:, :, 0]
        latent_B = latent[:, :, 1]
        # angles = 2 * torch.pi * (x[..., 0:1].unsqueeze(-1) * self.x_freq + x[...,1:2].unsqueeze(-1) * self.y_freq) 
        # angles = angles.unsqueeze(1)
        angles = 2 * torch.pi * (x[0:1,..., 0:1].unsqueeze(-1) * self.x_freq + x[0:1,...,1:2].unsqueeze(-1) * self.y_freq) 
    
        # print(latent_A.shape, latent_B.shape, angles.shape)
        # import pdb; pdb.set_trace()
        modulations = (latent_A * torch.cos(angles) - latent_B * torch.sin(angles)).reshape(x.shape[0], self.grid_channel, -1, self.grid_size*self.grid_size).sum(-1)

        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        # if len(x.shape)==3:
        #     x = x.unsqueeze(-2)
        #     modulations = grid_sample(latent, 2*x-1).squeeze(-1)
        #     # print(modulations.shape)
        # else:
        #     modulations = grid_sample(latent, 2*x-1)
        # modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenMoG(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        grid_base=64,
        grid_sum=True,
        share_grid=False
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.num_grid_base = grid_base
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        if share_grid: 
            self.grid_bases = nn.Parameter(torch.randn(1, self.num_grid_base, self.grid_size, self.grid_size))
        else:
            self.grid_bases = nn.Parameter(torch.randn(self.grid_channel, self.num_grid_base, self.grid_size, self.grid_size))
        self.latent_code = latent_dim
        self.grid_sum = grid_sum
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = self.num_grid_base * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)
        if self.grid_sum:
            latent_f = torch.sum(modulations_code.unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-3)
        else:
            latent_f = torch.mean(modulations_code.unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-3)
        # print(latent_f.shape)
        # latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenCodeMoG(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        grid_base=64,
        grid_sum=True,
        share_grid=False,
        share_code=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.num_grid_base = grid_base
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        if share_grid: 
            self.grid_bases = nn.Parameter(torch.randn(1, self.num_grid_base, self.grid_size, self.grid_size))
        else:
            self.grid_bases = nn.Parameter(torch.randn(self.grid_channel, self.num_grid_base, self.grid_size, self.grid_size))
        self.latent_code = latent_dim
        self.grid_sum = grid_sum
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        
        num_modulations = self.num_grid_base * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.share_code = share_code
            if self.share_code:
                self.modulation_net = LatentToModulation(
                    latent_dim,
                    (self.num_grid_base+dim_hidden) * self.grid_channel,
                    modulation_net_dim_hidden,
                    modulation_net_num_layers,
                )
            else:
                self.modulation_net = LatentToModulation(
                    latent_dim//2,
                    (dim_hidden) * self.grid_channel,
                    modulation_net_dim_hidden,
                    modulation_net_num_layers,
                )
                self.modulation_grid_net = LatentToModulation(
                    latent_dim//2,
                    (self.num_grid_base) * self.grid_channel,
                    modulation_net_dim_hidden,
                    modulation_net_num_layers,
                )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        if self.share_code:
            modulations_all = self.modulation_net(latent)
            modulations_f = modulations_all[:, :(self.num_grid_base) * self.grid_channel].reshape(x.shape[0], self.grid_channel, -1)
            modulations_code = modulations_all[:, (self.num_grid_base) * self.grid_channel:].reshape(x.shape[0], self.grid_channel, -1)
        else:
            modulations_code = self.modulation_net(latent[:,:int(0.5*latent.shape[-1])]).reshape(x.shape[0], self.grid_channel, -1)
            modulations_f = self.modulation_grid_net(latent[:,int(0.5*latent.shape[-1]):]).reshape(x.shape[0], self.grid_channel, -1)
        # print(modulations_code.shape, modulations_f.shape)
        if self.grid_sum:
            latent_f = torch.sum(modulations_f.unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-3)
        else:
            latent_f = torch.mean(modulations_f.unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-3)
        # print(latent_f.shape)
        # latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenMoGCodeChannel(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=1,
        siren_init=True,
        grid_base=64,
        grid_sum=True,
        share_grid=False
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.num_grid_base = grid_base
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        if share_grid: 
            self.grid_bases = nn.Parameter(torch.randn(1, self.num_grid_base, dim_hidden))
        else:
            self.grid_bases = nn.Parameter(torch.randn(self.grid_channel, self.num_grid_base, dim_hidden))
        self.latent_code = latent_dim
        self.grid_sum = grid_sum
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = self.num_grid_base * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)
        if self.grid_sum:
            latent_f = torch.sum(modulations_code.unsqueeze(-1) * self.grid_bases, dim=-2)
        else:
            latent_f = torch.mean(modulations_code.unsqueeze(-1) * self.grid_bases, dim=-2) # B, grid_channel, hidden
        # print(latent_f.shape)
        # latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # latent_f = latent_f.reshape(latent_f.shape[0],-1,self.grid_size,self.grid_size)
        # # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        # if len(x.shape)==3:
        #     # x = x.unsqueeze(-2)
        #     modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
        #     # print(modulations.shape)
        # else:
        #     modulations = grid_sample(latent_f, 2*x-1)
        # modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        modulations_f = latent_f 
        # modulations_f = modulations_f.view(modulations_f.shape[0], self.grid_channel, -1, modulations_f.shape[-1]).permute(0,1,3,2)
        # print(modulations_f.shape)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_f[:, idx:idx+1] + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_f[:, idx:idx+1]
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenMoGChannel(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        grid_base=64,
        grid_sum=True,
        share_grid=False
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.num_grid_base = grid_base
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        if share_grid: 
            self.grid_bases = nn.Parameter(torch.randn(1, self.num_grid_base, dim_hidden, self.grid_size, self.grid_size))
        else:
            self.grid_bases = nn.Parameter(torch.randn(self.grid_channel, self.num_grid_base, dim_hidden, self.grid_size, self.grid_size))
        print(self.grid_bases.shape)
        self.latent_code = latent_dim
        self.grid_sum = grid_sum
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = self.num_grid_base * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)
        if self.grid_sum:
            latent_f = torch.sum(modulations_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-4)
        else:
            latent_f = torch.mean(modulations_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-4)
        # print(latent_f.shape)
        # latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        latent_f = latent_f.reshape(latent_f.shape[0],-1,self.grid_size,self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        modulations_f = modulations_f.view(modulations_f.shape[0], self.grid_channel, -1, modulations_f.shape[-1]).permute(0,1,3,2)
        # print(modulations_f.shape)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_f[:, idx] + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_f[:, idx]
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenMoGChannelMS(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        grid_scale_ratio=1.5,
        siren_init=True,
        grid_base=64,
        grid_sum=True,
        share_grid=False
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.num_grid_base = grid_base
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        # if share_grid: 
        #     self.grid_bases = nn.Parameter(torch.randn(1, self.num_grid_base, dim_hidden, self.grid_size, self.grid_size))
        # else:
        #     self.grid_bases = nn.Parameter(torch.randn(self.grid_channel, self.num_grid_base, dim_hidden, self.grid_size, self.grid_size))

        self.grid_bases = nn.ParameterList(
            [nn.Parameter(torch.randn(self.num_grid_base, dim_hidden, int(self.grid_size*(grid_scale_ratio**i)), int(self.grid_size*(grid_scale_ratio**i)))) for i in range(self.grid_channel)]
            )


        
        self.latent_code = latent_dim
        self.grid_sum = grid_sum
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = self.num_grid_base * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)  # (b,grid_channel,grid_base)
        if self.grid_sum:
            # latent_f = torch.sum(modulations_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-4)
            latent_f = [torch.sum(modulations_code[:,i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.grid_bases[i], dim=-4) for i in range(self.grid_channel)]
        else:
            # latent_f = torch.mean(modulations_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-4)
            latent_f = [torch.mean(modulations_code[:,i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.grid_bases[i], dim=-4) for i in range(self.grid_channel)] # b,dim_hidden,grid_size,grid_size
        # import pdb; pdb.set_trace()
        # print(latent_f.shape)
        # latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # latent_f = latent_f.reshape(latent_f.shape[0],-1,self.grid_size,self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            # modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            modulations = [grid_sample(l_idx, 2*x.unsqueeze(-2)-1).squeeze(-1) for l_idx in latent_f]
            # print(modulations.shape)
        else:
            # modulations = grid_sample(latent_f, 2*x-1)
            modulations = [grid_sample(l_idx, 2*x-1) for l_idx in latent_f]
        modulations_f = [mod.view(mod.shape[0], mod.shape[1], -1).permute(0,2,1) for mod in modulations]
        # modulations_f = 
        # modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # modulations_f = modulations_f.view(modulations_f.shape[0], self.grid_channel, -1, modulations_f.shape[-1]).permute(0,1,3,2)
        # print(modulations_f.shape)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_f[idx] + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_f[idx]
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenMoGChannelV2(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        grid_base=64,
        grid_sum=True,
        grid_init_channel=True,
        share_grid=False
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.num_grid_base = grid_base
        self.grid_init_channel = grid_init_channel
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        if share_grid: 
            self.grid_bases = nn.Parameter(torch.randn(1, self.num_grid_base, self.grid_init_channel, self.grid_size, self.grid_size))
        else:
            self.grid_bases = nn.Parameter(torch.randn(self.grid_channel, self.num_grid_base, self.grid_init_channel, self.grid_size, self.grid_size))
        self.latent_code = latent_dim
        self.grid_sum = grid_sum
        self.linear_channel = nn.Linear(self.grid_init_channel, dim_hidden)
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = self.num_grid_base * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)
        if self.grid_sum:
            latent_f = torch.sum(modulations_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-4)
        else:
            latent_f = torch.mean(modulations_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.grid_bases, dim=-4)
        # print(latent_f.shape)
        # latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # latent_f = latent_f.reshape(latent_f.shape[0],-1,self.grid_size,self.grid_size)
        latent_f = self.linear_channel(latent_f.permute(0,1,3,4,2)).permute(0,1,4,2,3).reshape(latent_f.shape[0],-1,self.grid_size,self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        modulations_f = modulations_f.view(modulations_f.shape[0], self.grid_channel, -1, modulations_f.shape[-1]).permute(0,1,3,2)
        # print(modulations_f.shape)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_f[:, idx] + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_f[:, idx]
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenCodeGrids(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenCodeGridsMS(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        num_scales=4,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.num_scales =num_scales
        self.pool = torch.nn.AvgPool2d(2, 2)
        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            for _ in range(self.num_scales-1):
                latent_f = self.pool(latent_f)
                modulations = modulations + grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            modulations = modulations / self.num_scales
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
            for _ in range(self.num_scales-1):
                latent_f = self.pool(latent_f)
                modulations = modulations + grid_sample(latent_f, 2*x-1)
            modulations = modulations / self.num_scales
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenCodeGridsMSC(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        num_scales=4,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.num_scales =num_scales
        self.pool = torch.nn.AvgPool2d(3, 1, 1)
        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            for _ in range(self.num_scales-1):
                latent_f = self.pool(latent_f)
                modulations = modulations + grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            modulations = modulations / self.num_scales
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
            for _ in range(self.num_scales-1):
                latent_f = self.pool(latent_f)
                modulations = modulations + grid_sample(latent_f, 2*x-1)
            modulations = modulations / self.num_scales
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])
    
class ModulatedSirenCodeGridsNorm(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale_code = modulations_code[:, idx:idx+1] + 1.0
                scale_f = modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale_code = 1.0
                scale_f = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift_code = modulations_code[:, idx:idx+1] 
                shift_f = modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift_f = 0.0
                shift_code = 0.0 

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale_code * x + shift_code  # Broadcast scale and shift across num_points
            x = x - torch.mean(x, dim=-1, keepdim=True)
            x = scale_f * x + shift_f 
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenCodeShiftGridsScale(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        # if self.modulate_scale and self.modulate_shift:
        #     self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        # if self.modulate_scale and self.modulate_shift:
        #     # If we modulate both scale and shift, we have twice the number of
        #     # modulations at every layer and feature
        #     num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            # if self.modulate_scale:
            #     # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
            #     # modulations remain zero centered
            #     scale_code = modulations_code[:, idx:idx+1] + 1.0
            #     scale_f = modulations_f[:, idx].unsqueeze(2) + 1.0
            #     idx += 1
            # else:
            #     scale_code = 1.0
            #     scale_f = 1.0
            scale_f = modulations_f[:, idx].unsqueeze(2) + 1.0
            shift_code = modulations_code[:, idx:idx+1] 
            idx += 1
            # if self.modulate_shift:
            #     # Shape (batch_size, 1, dim_hidden)
            #     shift_code = modulations_code[:, idx:idx+1] 
            #     shift_f = modulations_f[:, idx].unsqueeze(2)
            #     idx += 1
            # else:
            #     shift_f = 0.0
            #     shift_code = 0.0 

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale_f * x + shift_code  # Broadcast scale and shift across num_points
           
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenCodeLastGrids(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = 1
        self.code_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
            self.code_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.code_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        # import pdb; pdb.set_trace()
        idx = 0
        idx_f = 0
        for layer_id, module in enumerate(self.net):
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                # scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2) + 1.0
                if layer_id == len(self.net)-1:
                    scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx_f].unsqueeze(2) + 1.0
                    idx_f += 1
                else:
                    scale = modulations_code[:, idx:idx+1] + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                # shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2)
                if layer_id == len(self.net)-1:
                    # shift += modulations_f[:, idx_f].unsqueeze(2)
                    shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx_f].unsqueeze(2)
                    idx_f += 1
                else:
                    shift = modulations_code[:, idx:idx+1] 
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenSCodeGrids(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
  
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_code[:, idx:idx+1].unsqueeze(2) + modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_code[:, idx:idx+1].unsqueeze(2) + modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenCodeFourierC(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        cos_term = torch.cos(angles)
        sin_term = torch.sin(angles)
        self.register_buffer('cos_term', cos_term)
        self.register_buffer('sin_term', sin_term)
        self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim-2 * self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - 2 * self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, 2, 1, self.grid_size, self.grid_size)
        latent_A = latent_f[:, :, 0]
        latent_B = latent_f[:, :, 1]
        # angles = 2 * torch.pi * (x[..., 0:1].unsqueeze(-1) * self.x_freq + x[...,1:2].unsqueeze(-1) * self.y_freq) 
        # angles = angles.unsqueeze(1)
        angles = 2 * torch.pi * (x[0:1,..., 0:1].unsqueeze(-1) * self.x_freq + x[0:1,...,1:2].unsqueeze(-1) * self.y_freq) 
    
        # print(latent_A.shape, latent_B.shape, angles.shape)
        # import pdb; pdb.set_trace()
        # modulations_f = (latent_A * torch.cos(angles) - latent_B * torch.sin(angles)).reshape(x.shape[0], self.grid_channel, -1, self.grid_size*self.grid_size).sum(-1)
        modulations_f =(latent_A * torch.cos(angles+latent_B)).reshape(x.shape[0], self.grid_channel, -1, self.grid_size*self.grid_size).sum(-1)

        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        # if len(x.shape)==3:
        #     x = x.unsqueeze(-2)
        #     modulations = grid_sample(latent, 2*x-1).squeeze(-1)
        #     # print(modulations.shape)
        # else:
        #     modulations = grid_sample(latent, 2*x-1)
        # modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenCode2FourierC(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        # for inverse fourier transform
        m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        cos_term = torch.cos(angles)
        sin_term = torch.sin(angles)
        self.register_buffer('cos_term', cos_term)
        self.register_buffer('sin_term', sin_term)
        self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        # self.x_freq = torch.nn.parameter.Parameter(torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.y_freq = torch.nn.parameter.Parameter(torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim-2 * self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                2 * self.grid_channel * grid_size * grid_size,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent_f = self.modulation_net(latent).reshape(x.shape[0], self.grid_channel, 2, 1, self.grid_size, self.grid_size)

        # latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, 2, 1, self.grid_size, self.grid_size)
        latent_A = latent_f[:, :, 0]
        latent_B = latent_f[:, :, 1]
        # angles = 2 * torch.pi * (x[..., 0:1].unsqueeze(-1) * self.x_freq + x[...,1:2].unsqueeze(-1) * self.y_freq) 
        # angles = angles.unsqueeze(1)
        angles = 2 * torch.pi * (x[0:1,..., 0:1].unsqueeze(-1) * self.x_freq + x[0:1,...,1:2].unsqueeze(-1) * self.y_freq) 
    
        # print(latent_A.shape, latent_B.shape, angles.shape)
        # import pdb; pdb.set_trace()
        # modulations_f = (latent_A * torch.cos(angles) - latent_B * torch.sin(angles)).reshape(x.shape[0], self.grid_channel, -1, self.grid_size*self.grid_size).sum(-1)
        modulations_f = (latent_A * torch.cos(angles+latent_B)).reshape(x.shape[0], self.grid_channel, -1, self.grid_size*self.grid_size).mean(-1)

        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        # if len(x.shape)==3:
        #     x = x.unsqueeze(-2)
        #     modulations = grid_sample(latent, 2*x-1).squeeze(-1)
        #     # print(modulations.shape)
        # else:
        #     modulations = grid_sample(latent, 2*x-1)
        # modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsChannel(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = dim_hidden * (num_layers - 1)
        self.dim_hidden = dim_hidden
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.in_channel = latent_dim // (self.grid_size * self.grid_size)
        self.modulation_net = torch.nn.Conv2d(self.in_channel, self.grid_channel, kernel_size=1)

        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.in_channel, self.grid_size, self.grid_size)
        latent = self.modulation_net(latent)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1).permute(0,2,1)
        modulations = modulations.reshape(modulations.shape[0], modulations.shape[1], -1, self.dim_hidden)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, :, idx] + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, :, idx]
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsInit(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.grid_init = nn.Parameter(torch.zeros(1, self.grid_channel, self.grid_size, self.grid_size))
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = self.grid_init + latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGrids(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        # self.grid_init = nn.Parameter(torch.zeros(1, self.grid_channel, self.grid_size, self.grid_size))
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])
    
class ModulatedSirenGridsMS(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        num_scales=4,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.num_scales =num_scales
        self.pool = torch.nn.AvgPool2d(2, 2)
        # self.grid_init = nn.Parameter(torch.zeros(1, self.grid_channel, self.grid_size, self.grid_size))
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            for _ in range(self.num_scales-1):
                latent = self.pool(latent)
                modulations = modulations + grid_sample(latent, 2*x-1).squeeze(-1)
            modulations = modulations / self.num_scales
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
            for _ in range(self.num_scales-1):
                latent = self.pool(latent)
                modulations = modulations + grid_sample(latent, 2*x-1)
            modulations = modulations / self.num_scales
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsMSC(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        num_scales=4,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.num_scales =num_scales
        self.pool = torch.nn.AvgPool2d(3, 1, 1)
        # self.grid_init = nn.Parameter(torch.zeros(1, self.grid_channel, self.grid_size, self.grid_size))
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            for _ in range(self.num_scales-1):
                latent = self.pool(latent)
                modulations = modulations + grid_sample(latent, 2*x-1).squeeze(-1)
            modulations = modulations / self.num_scales
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
            for _ in range(self.num_scales-1):
                latent = self.pool(latent)
                modulations = modulations + grid_sample(latent, 2*x-1)
            modulations = modulations / self.num_scales
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsSSGB(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        num_scales=4,
        gaussian_sigma=0.85,
        gaussian_kernel_size=3,
        gaussian_stride=1,
        gaussian_padding=1,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.num_scales =num_scales
        # self.pool = torch.nn.AvgPool2d(3, 1, 1)
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_stride = gaussian_stride
        self.gaussian_padding= gaussian_padding
        kernels = gaussian_kernel(gaussian_kernel_size, gaussian_sigma)
        self.register_buffer('gaussian_kernel', kernels.reshape(1,1,gaussian_kernel_size,gaussian_kernel_size).repeat(self.grid_channel,1,1,1))
        # self.grid_init = nn.Parameter(torch.zeros(1, self.grid_channel, self.grid_size, self.grid_size))
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        for _ in range(self.num_scales-1):
            latent = F.conv2d(latent, self.gaussian_kernel, stride=self.gaussian_stride, padding=self.gaussian_padding, groups=self.grid_channel)

        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
        else:
            modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsMSGB(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
        num_scales=4,
        gaussian_sigma=0.85,
        gaussian_kernel_size=3,
        gaussian_stride=1,
        gaussian_padding=1,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.num_scales =num_scales
        # self.pool = torch.nn.AvgPool2d(3, 1, 1)
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_stride = gaussian_stride
        self.gaussian_padding= gaussian_padding
        kernels = gaussian_kernel(gaussian_kernel_size, gaussian_sigma)
        self.register_buffer('gaussian_kernel', kernels.reshape(1,1,gaussian_kernel_size,gaussian_kernel_size).repeat(self.grid_channel,1,1,1))
        # self.grid_init = nn.Parameter(torch.zeros(1, self.grid_channel, self.grid_size, self.grid_size))
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            for _ in range(self.num_scales-1):
                # latent = self.pool(latent)
                latent = F.conv2d(latent, self.gaussian_kernel, stride=self.gaussian_stride, padding=self.gaussian_padding, groups=self.grid_channel)
                modulations = modulations + grid_sample(latent, 2*x-1).squeeze(-1)
            modulations = modulations / self.num_scales
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
            for _ in range(self.num_scales-1):
                # latent = self.pool(latent)
                latent = F.conv2d(latent, self.gaussian_kernel, stride=self.gaussian_stride, padding=self.gaussian_padding, groups=self.grid_channel)
                modulations = modulations + grid_sample(latent, 2*x-1)
            modulations = modulations / self.num_scales
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenFields(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        self.modulation_net = ModulatedSiren(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=self.grid_channel,
            num_layers=modulation_net_num_layers,
            w0=30,
            w0_initial=30,
            use_bias=use_bias,
            modulate_scale=False,
            modulate_shift=True,
            use_latent=use_latent,
            latent_dim=latent_dim,
            modulation_net_dim_hidden=128,
            modulation_net_num_layers=1,
            last_activation=last_activation,
            siren_init=True,
        )
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        # latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        # if len(x.shape)==3:
        #     x = x.unsqueeze(-2)
        #     modulations = grid_sample(latent, 2*x-1).squeeze(-1)
        #     # print(modulations.shape)
        # else:
        #     modulations = grid_sample(latent, 2*x-1)
        # modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)

        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])
        modulations = self.modulation_net.modulated_forward(x, latent)
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[...,idx:idx+1] + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[...,idx:idx+1]
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenFieldsChannel(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = dim_hidden * (num_layers - 1)
        self.dim_hidden = dim_hidden
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm

        self.modulation_net = ModulatedSiren(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=self.grid_channel,
            num_layers=num_layers,
            w0=w0,
            w0_initial=w0_initial,
            use_bias=use_bias,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
            use_latent=use_latent,
            latent_dim=latent_dim,
            modulation_net_dim_hidden=modulation_net_dim_hidden,
            modulation_net_num_layers=modulation_net_num_layers,
            last_activation=last_activation,
        )
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        # latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        # if len(x.shape)==3:
        #     x = x.unsqueeze(-2)
        #     modulations = grid_sample(latent, 2*x-1).squeeze(-1)
        #     # print(modulations.shape)
        # else:
        #     modulations = grid_sample(latent, 2*x-1)
        # modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)

        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])
        modulations = self.modulation_net.modulated_forward(x, latent)
        modulations = modulations.reshape(modulations.shape[0], modulations.shape[1], -1, self.dim_hidden)
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:,:,idx] + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:,:,idx]
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsGlobal(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.modulation_net = torch.nn.Conv1d(1, dim_hidden, kernel_size=1)

        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        global_modulations = self.modulation_net(torch.mean(latent.reshape(x.shape[0], self.grid_channel, -1), dim=-1, keepdim=True).permute(0,2,1)).permute(0,2,1)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                global_shift = global_modulations[:, idx:idx+1]
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift + global_shift # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsSet(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations = latent.reshape(x.shape[0], self.grid_channel, self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        # if len(x.shape)==3:
        #     x = x.unsqueeze(-2)
        #     modulations = grid_sample(latent, 2*x-1).squeeze(-1)
        #     # print(modulations.shape)
        # else:
        #     modulations = grid_sample(latent, 2*x-1)
        # modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

class ModulatedSirenGridsLR(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.rank = latent_dim // self.grid_channel // self.grid_size // 2
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, 2, self.grid_size, self.rank)
        A = latent[:, :, 0]
        B = latent[:, :, 1]
        latent = torch.matmul(A, B.permute(0,1,3,2))
        # print(latent.shape)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            x = x.unsqueeze(-2)
            modulations = grid_sample(latent, 2*x-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])
    
class ModulatedSirenGridsZeroMean(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            # if self.use_norm:
            #     x = nn.functional.instance_norm(x)
            x = x - torch.mean(x, dim=-1, keepdim=True)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])
    
class ModulatedSirenOneGrids(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        '''
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        
        for module in self.net:
            idx = 0
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ModulatedSirenOneGridsConv(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = 1
        # if self.modulate_scale and self.modulate_shift:
        #     self.grid_channel *= 2
        self.use_norm = use_norm
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        self.modulation_net = ConvLatentToModulation(
                input_dim=2,
                grid_size=self.grid_size,
                latent_dim=self.grid_channel,
                num_modulations=num_modulations,
                kernel_size=3
            )
        
        '''
        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        '''

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        latent = latent.reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        # print(latent.shape)
        latent = self.modulation_net(latent)
        # print(latent.shape)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        modulations = grid_sample(latent, 2*x-1)
        modulations = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

 
class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self, latent_dim, num_modulations, dim_hidden, num_layers, activation=nn.SiLU
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation = activation

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden),
                               self.activation()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size), requires_grad=True)
        # Add latent_dim attribute for compatibility with LatentToModulation model
        self.latent_dim = size

    def forward(self, x):
        return x + self.bias


class ConvLatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.

    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self, input_dim, grid_size, latent_dim, num_modulations, kernel_size=1, padding=1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.grid_size = grid_size

        if input_dim == 2:
            self.net = nn.Conv2d(latent_dim, num_modulations, kernel_size, padding=padding)
        elif input_dim == 1:
            self.net = nn.Conv1d(latent_dim, num_modulations, kernel_size)
            # self.coords = shape2coordinates([grid_size, grid_size])

    def forward(self, latent):
        return self.net(latent)


class ConvModulatedSiren(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        mu=0,
        sigma=1,
        last_activation=None,
        grid_size=8,
        interpolation="bilinear",
        conv_kernel=3,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.grid_size = grid_size
        self.interpolation = interpolation
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        self.modulation_net = ConvLatentToModulation(
            dim_in, grid_size, latent_dim, num_modulations, conv_kernel
        )

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations

        # self.mode = "linear" if dim_in == 1 else "bilinear"

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)
        # print('mod1', modulations.shape)
        # v1 modulations = torch.nn.functional.grid_sample(modulations, x, mode='bilinear', padding_mode='zeros', align_corners=None)
        # v2 modulations = grid_sample(modulations, x)
        # vtest
        modulations = torch.nn.functional.interpolate(
            modulations, x.shape[1:-1], mode=self.interpolation
        )
        # print('mod2', modulations.shape)

        x = x.view(x.shape[0], -1, x.shape[-1])
        # modulations = modulations.permute(0, 2, 3, 1)

        # place the channel at the end
        modulations = torch.movedim(modulations, 1, -1)
        modulations = modulations.view(x.shape[0], -1, self.num_modulations)
        # print('mod', modulations.shape)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, :, idx: idx + self.dim_hidden] + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, :, mid_idx + idx: mid_idx + idx + self.dim_hidden
                ]
            else:
                shift = 0.0

            x = module.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        # print('out', out.shape)
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class ConvModulatedSiren2(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        mu=0,
        sigma=1,
        last_activation=None,
        grid_size=8,
        interpolation="bilinear",
        conv_kernel=3,
        rescale_coordinate=False,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.grid_size = grid_size
        self.interpolation = interpolation
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.rescale_coordinate = rescale_coordinate

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        self.modulation_net = ConvLatentToModulation(
            dim_in, grid_size, latent_dim, num_modulations, conv_kernel
        )

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations

        # create modulation positions
        crds = []
        for i in range(dim_in):
            crds.append(torch.linspace(0.0, 1.0, grid_size))
        self.modulation_grid = torch.stack(
            torch.meshgrid(*crds, indexing="ij"), dim=-1)

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        batch_size, num_points = x.shape[0], x.shape[1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # print('toto', modulations.shape, latent.shape)
        modulations = torch.movedim(modulations, 1, -1)

        modulations = modulations.view(
            modulations.shape[0], -1, modulations.shape[-1])

        mod_grid = self.modulation_grid.view(-1, x.shape[-1])
        mod_grid = (
            mod_grid.unsqueeze(0)
            .repeat(x_shape[0], *(1,) * self.modulation_grid.ndim)
            .cuda()
        )

        x_batch = (
            torch.arange(modulations.shape[0])
            .view(-1, 1)
            .repeat(1, modulations.shape[1])
            .flatten()
            .cuda()
        )
        y_batch = (
            torch.arange(x.shape[0]).view(-1, 1).repeat(1,
                                                        x.shape[1]).flatten().cuda()
        )

        # print('tototot')
        # print(einops.rearrange(modulations, 'b d c -> (b d) c').shape,
        #      einops.rearrange(mod_grid, 'b d c -> (b d) c').shape,
        #      x.view(-1, x.shape[-1]).shape,
        #      x_batch.shape,
        #      y_batch.shape)

        modulations = knn_interpolate_custom(
            einops.rearrange(modulations, "b d c -> (b d) c"),
            einops.rearrange(mod_grid, "b d c -> (b d) c"),
            x.view(-1, x.shape[-1]),
            batch_x=x_batch,
            batch_y=y_batch,
        )
        if self.rescale_coordinate:
            x = rescale_coordinate(
                self.modulation_grid.cuda(), x.view(-1, x.shape[-1]))

        x = x.view(batch_size, -1, x.shape[-1])
        modulations = modulations.view(batch_size, -1, modulations.shape[-1])

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, :, idx: idx + self.dim_hidden] + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[
                    :, :, mid_idx + idx: mid_idx + idx + self.dim_hidden
                ]
            else:
                shift = 0.0

            x = module.linear(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        # print('out', out.shape)
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class SSN(nn.Module):
    """Simple Sine model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.w0 = w0

        layers = [nn.Linear(dim_in, dim_hidden)]
        for j in range(self.num_layers-1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(nn.Linear(dim_hidden, dim_out))
        self.layers = nn.ModuleList(layers)
        
        self.init_weights()
                          
    def init_weights(self):
        for j, layer in enumerate(self.layers):
            if j == 0:
                nn.init.normal(layer.weight, 0,  np.sqrt(2) / np.sqrt(layer.weight.shape[1]))
                #nn.init.uniform(layer.weight, -self.w0 / layer.weight.shape[1], self.w0 / layer.weight.shape[1])
            else:
                nn.init.normal(layer.weight, 0, np.sqrt(2) / np.sqrt(layer.weight.shape[1]))
            print(layer.weight.std())
        
    def forward(self, x):
        for j, layer in enumerate(self.layers[:-1]):
            if j == 0:
                x = torch.sin(self.w0*layer(x))
            else:
                x = torch.sin(layer(x))
        out = self.layers[-1](x)
        return out 



class ModulatedSSN(nn.Module):
    """Modulated Sine model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.w0 = w0
        self.latent_dim = latent_dim    

        layers = [nn.Linear(dim_in, dim_hidden)]
        for j in range(self.num_layers-1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(nn.Linear(dim_hidden, dim_out))
        self.layers = nn.ModuleList(layers)
        self.num_modulations = dim_hidden * len(self.layers[:-1])    

        self.modulation_net = LatentToModulation(
                latent_dim,
                self.num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        self.init_weights()
                          
    def init_weights(self):
        for j, layer in enumerate(self.layers):
            if j == 0:
                nn.init.normal(layer.weight, 0,  np.sqrt(2) / np.sqrt(layer.weight.shape[1]))
                #nn.init.uniform(layer.weight, -self.w0 / layer.weight.shape[1], self.w0 / layer.weight.shape[1])
            else:
                nn.init.normal(layer.weight, 0, np.sqrt(2) / np.sqrt(layer.weight.shape[1]))
            print(layer.weight.std())
        
    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        modulations = self.modulation_net(z)
        modulations = modulations.reshape(-1, self.latent_dim, len(self.layers[:-1])).unsqueeze(1)

        for j, layer in enumerate(self.layers[:-1]):
            if j == 0:
                x = torch.sin(self.w0*layer(x) + modulations[..., j])
            else:
                x = torch.sin(layer(x) + modulations[..., j])
        out = self.layers[-1](x)

        return out.view(*x_shape, out.shape[-1])


class CorssAttention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        N_q = query.shape[1]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 2,B,num_head,N,head_dim
        k, v = kv.unbind(0)
        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0,2,1,3)
        q, k = self.q_norm(q), self.k_norm(k)
        # import pdb; pdb.set_trace()
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # B,num_head,N_q,N
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v    # B,num_head,N_q,head_dim

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class NeRFEncoding(nn.Module):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers."""

    def __init__(
        self,
        num_freq,
        max_freq_log2,
        log_sampling=True,
        include_input=True,
        input_dim=3,
        base_freq=2,
    ):
        """Initialize the module.
        Args:
            num_freq (int): The number of frequency bands to sample.
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.
        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        self.base_freq = base_freq

        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            self.bands = self.base_freq ** torch.linspace(
                0.0, max_freq_log2, steps=num_freq
            )
        else:
            self.bands = torch.linspace(
                1, self.base_freq**max_freq_log2, steps=num_freq
            )

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(self, coords, with_batch=True):
        """Embeds the coordinates.
        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]
        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        if with_batch:
            N = coords.shape[0]
            winded = (coords[..., None, :] * self.bands[None,None,:,None]).reshape(
                N, coords.shape[1], coords.shape[-1] * self.num_freq)
            encoded = torch.cat([torch.sin(2*np.pi*winded), torch.cos(2*np.pi*winded)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)

        else:
            N = coords.shape[0]
            winded = (coords[:, None] * self.bands[None, :, None]).reshape(
                N, coords.shape[1] * self.num_freq
            )
            encoded = torch.cat([torch.sin(2*np.pi*winded), torch.cos(2*np.pi*winded)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    # def name(self) -> str:
    #     """A human readable name for the given wisp module."""
    #     return "Positional Encoding"

    # def public_properties(self) -> Dict[str, Any]:
    #     """Wisp modules expose their public properties in a dictionary.
    #     The purpose of this method is to give an easy table of outwards facing attributes,
    #     for the purpose of logging, gui apps, etc.
    #     """
    #     return {
    #         "Output Dim": self.out_dim,
    #         "Num. Frequencies": self.num_freq,
    #         "Max Frequency": f"2^{self.max_freq_log2}",
    #         "Include Input": self.include_input,
    #     }

class ModulatedSirenCA(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_coords,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.N_token = latent_dim // dim_in
        self.modulation_net = CorssAttention(
            dim=dim_in,
            num_heads=4,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
        )
        self.embedding = NeRFEncoding(
            num_freq=dim_in//(2*dim_coords),
            max_freq_log2=6,
            log_sampling=True,
            include_input=False,
            input_dim=dim_coords,
            base_freq=2,
        )

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

        x = x.view(x.shape[0], -1, x.shape[-1])
        x_embedding = self.embedding(x)
        # print(x_embedding.shape)
        # import pdb; pdb.set_trace()
        latent = latent.view(latent.shape[0], self.N_token, -1)
        # x: torch.Size([128, 4096, 2])
        # latent: torch.Size([128, 128])
        # modulations: torch.Size([128, 384])
        # Shape (batch_size, num_modulations)
        x = self.modulation_net(x_embedding, latent)
        
        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        # idx = 0
        for module in self.net:
            # if self.modulate_scale:
            #     # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
            #     # modulations remain zero centered
            #     scale = modulations[:, idx: idx +
            #                         self.dim_hidden].unsqueeze(1) + 1.0
            # else:
            #     scale = 1.0

            # if self.modulate_shift:
            #     # Shape (batch_size, 1, dim_hidden)
            #     shift = modulations[
            #         :, mid_idx + idx: mid_idx + idx + self.dim_hidden
            #     ].unsqueeze(1)
            # else:
            #     shift = 0.0

            x = module.linear(x)
            # x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class IPC(nn.Module):

    def __init__(
        self,
        freq_dim_in,
        freq_dim_out,
        composer_dim_out,
        activation=None,
    ):
        super().__init__()
        self.freq_layer = nn.Linear(freq_dim_in, freq_dim_out)
        self.freq_act = Sine(1) if activation is None else activation
        self.composer_dim_out = composer_dim_out
        self.bias = 1.0 / freq_dim_out
    
    def forward(self, x, latent):
        freq = self.freq_act(self.freq_layer(x))    # (batch_size, -1, freq_dim_out)
        latent = latent.reshape(latent.shape[0], 1, self.composer_dim_out, -1)
        modulations = (latent+self.bias) @ freq.unsqueeze(-1)
        return modulations.squeeze(-1)  # (batch_size, -1, self.composer_dim_out) 

class ModulatedSirenIPC(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_coords,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        # self.N_token = latent_dim // dim_in
        
        self.embedding = NeRFEncoding(
            num_freq=64,
            max_freq_log2=6,
            log_sampling=True,
            include_input=False,
            input_dim=dim_coords,
            base_freq=2,
        )

        self.modulation_net = IPC(
            freq_dim_in=self.embedding.out_dim,
            freq_dim_out=latent_dim // dim_in,
            composer_dim_out=dim_in,
        )

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

        x = x.view(x.shape[0], -1, x.shape[-1])
        x_embedding = self.embedding(x)
        # print(x_embedding.shape)
        # import pdb; pdb.set_trace()
        # latent = latent.view(latent.shape[0], self.N_token, -1)
        # x: torch.Size([128, 4096, 2])
        # latent: torch.Size([128, 128])
        # modulations: torch.Size([128, 384])
        # Shape (batch_size, num_modulations)
        x = self.modulation_net(x_embedding, latent)
        
        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        # idx = 0
        # for module in self.net:
        #     # if self.modulate_scale:
        #     #     # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
        #     #     # modulations remain zero centered
        #     #     scale = modulations[:, idx: idx +
        #     #                         self.dim_hidden].unsqueeze(1) + 1.0
        #     # else:
        #     #     scale = 1.0

        #     # if self.modulate_shift:
        #     #     # Shape (batch_size, 1, dim_hidden)
        #     #     shift = modulations[
        #     #         :, mid_idx + idx: mid_idx + idx + self.dim_hidden
        #     #     ].unsqueeze(1)
        #     # else:
        #     #     shift = 0.0

        #     x = module.linear(x)
        #     # x = scale * x + shift  # Broadcast scale and shift across num_points
        #     x = module.activation(x)  # (batch_size, num_points, dim_hidden)

        #     # idx = idx + self.dim_hidden
        x = self.net(x)
        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class IPCL(nn.Module):

    def __init__(
        self,
        freq_dim_in,
        freq_dim_out,
        composer_dim_out,
        activation=None,
    ):
        super().__init__()
        self.freq_layer = nn.Linear(freq_dim_in, freq_dim_out)
        self.freq_act = Sine(1) if activation is None else activation
        self.composer_dim_out = composer_dim_out
        # bias = torch.ones(composer_dim_out, freq_dim_out) * 1.0 / freq_dim_out
        self.bias = nn.Parameter(torch.ones(composer_dim_out, freq_dim_out) * 1.0 / freq_dim_out)
    
    def forward(self, x, latent):
        freq = self.freq_act(self.freq_layer(x))    # (batch_size, -1, freq_dim_out)
        latent = latent.reshape(latent.shape[0], 1, self.composer_dim_out, -1)
        modulations = (latent+self.bias) @ freq.unsqueeze(-1)
        # import pdb; pdb.set_trace()
        return modulations.squeeze(-1)  # (batch_size, -1, self.composer_dim_out) 
class ModulatedSirenIPCL(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_coords,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        # self.N_token = latent_dim // dim_in
        
        self.embedding = NeRFEncoding(
            num_freq=64,
            max_freq_log2=6,
            log_sampling=True,
            include_input=False,
            input_dim=dim_coords,
            base_freq=2,
        )

        self.modulation_net = IPCL(
            freq_dim_in=self.embedding.out_dim,
            freq_dim_out=latent_dim // dim_in,
            composer_dim_out=dim_in,
        )

    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)

        x = x.view(x.shape[0], -1, x.shape[-1])
        x_embedding = self.embedding(x)
        # print(x_embedding.shape)
        # import pdb; pdb.set_trace()
        # latent = latent.view(latent.shape[0], self.N_token, -1)
        # x: torch.Size([128, 4096, 2])
        # latent: torch.Size([128, 128])
        # modulations: torch.Size([128, 384])
        # Shape (batch_size, num_modulations)
        x = self.modulation_net(x_embedding, latent)
        
        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        # idx = 0
        # for module in self.net:
        #     # if self.modulate_scale:
        #     #     # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
        #     #     # modulations remain zero centered
        #     #     scale = modulations[:, idx: idx +
        #     #                         self.dim_hidden].unsqueeze(1) + 1.0
        #     # else:
        #     #     scale = 1.0

        #     # if self.modulate_shift:
        #     #     # Shape (batch_size, 1, dim_hidden)
        #     #     shift = modulations[
        #     #         :, mid_idx + idx: mid_idx + idx + self.dim_hidden
        #     #     ].unsqueeze(1)
        #     # else:
        #     #     shift = 0.0

        #     x = module.linear(x)
        #     # x = scale * x + shift  # Broadcast scale and shift across num_points
        #     x = module.activation(x)  # (batch_size, num_points, dim_hidden)

        #     # idx = idx + self.dim_hidden
        x = self.net(x)
        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


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


class GridToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self, grid_channel, grid_size
    ):
        super().__init__()
        self.grid_channel = grid_channel
        self.grid_size = grid_size
        self.width = grid_channel
        modes = grid_size // 2 + 1 if grid_size // 2 + 1 < 12 else 12 

        self.modes1 = modes # grid_size // 2 + 1 # 12 
        self.modes2 = modes # grid_size // 2 + 1 # 12 
        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.grid_init = nn.Parameter(torch.zeros(1, self.grid_channel, self.grid_size, self.grid_size))

    def forward(self, latent):
        x = self.grid_init + latent
        # x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        # x = x.permute(0, 1, 2, 3)
        return x


class GridToModulationLayers(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self, grid_channel, grid_size
    ):
        super().__init__()
        self.grid_channel = grid_channel
        self.grid_size = grid_size
        self.width = grid_channel
        self.modes1 = grid_size // 2 + 1
        self.modes2 = grid_size // 2 + 1 # 12 
        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.conv1 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.grid_init = nn.Parameter(torch.zeros(1, self.grid_channel, self.grid_size, self.grid_size))

    def forward(self, latent):
        x = self.grid_init + latent
        # x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        # x = x.permute(0, 1, 2, 3)
        return x
    

class ModulatedSirenCodeGridsFNO(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.grid_modulation_net = GridToModulation(self.grid_channel, self.grid_size)
        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        latent_f = self.grid_modulation_net(latent_f)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])
    
class ModulatedSirenCodeGridsFNO2L(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        use_norm=False,
        grid_size=64,
        siren_init=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
            siren_init=siren_init,
        )
        # Must modulate at least one of scale and shift
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
        self.grid_size = grid_size 
        self.grid_channel = num_layers - 1
        if self.modulate_scale and self.modulate_shift:
            self.grid_channel *= 2
        self.use_norm = use_norm
        self.grid_modulation_net = GridToModulationLayers(self.grid_channel, self.grid_size)
        # for inverse fourier transform
        # m = torch.arange(grid_size, dtype=torch.float32).view(grid_size, 1, 1, 1)
        # n = torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1, 1)
        # k = torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size, 1)
        # l = torch.arange(grid_size, dtype=torch.float32).view(1, 1, 1, grid_size)
        # angles = 2 * torch.pi * ((m * k / grid_size) + (n * l / grid_size)) 
        # cos_term = torch.cos(angles)
        # sin_term = torch.sin(angles)
        # self.register_buffer('cos_term', cos_term)
        # self.register_buffer('sin_term', sin_term)
        # self.register_buffer('x_freq', torch.arange(grid_size, dtype=torch.float32).view(1, grid_size, 1))
        # self.register_buffer('y_freq', torch.arange(grid_size, dtype=torch.float32).view(1, 1, grid_size))
        self.latent_code = latent_dim - self.grid_channel * grid_size * grid_size
        # self.modulation_net = LatentToModulation(
        #         latent_dim,
        #         2 * self.grid_channel * grid_size * grid_size,
        #         modulation_net_dim_hidden,
        #         modulation_net_num_layers,
        #     )
        
        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated
        num_modulations = dim_hidden * (num_layers - 1)
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim - self.grid_channel * grid_size * grid_size,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        modulations_code = self.modulation_net(latent[:,:self.latent_code]).reshape(x.shape[0], self.grid_channel, -1)

        latent_f = latent[:,self.latent_code:].reshape(x.shape[0], self.grid_channel, self.grid_size, self.grid_size)
        latent_f = self.grid_modulation_net(latent_f)
        # modulations = nn.functional.grid_sample(latent, 2*x-1, padding_mode='border', align_corners=True)# .reshape(x.shape[0],self.grid_channel, x.shape[1],x.shape[2])
        if len(x.shape)==3:
            # x = x.unsqueeze(-2)
            modulations = grid_sample(latent_f, 2*x.unsqueeze(-2)-1).squeeze(-1)
            # print(modulations.shape)
        else:
            modulations = grid_sample(latent_f, 2*x-1)
        modulations_f = modulations.view(modulations.shape[0], modulations.shape[1], -1)
        # print(latent.shape, modulations.shape)
        # import pdb; pdb.set_trace()
        # Shape (batch_size, num_modulations)
        # modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        # mid_idx = (
        #     self.num_modulations // 2
        #     if (self.modulate_scale and self.modulate_shift)
        #     else 0
        # )
        idx = 0
        for module in self.net:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2) + 1.0
                idx += 1
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations_code[:, idx:idx+1] + modulations_f[:, idx].unsqueeze(2)
                idx += 1
            else:
                shift = 0.0

            x = module.linear(x)
            if self.use_norm:
                x = nn.functional.instance_norm(x)
            x = scale * x + shift  # Broadcast scale and shift across num_points
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])

def gaussian_kernel(size: int, sigma: float):
    """
    Generates a 2D Gaussian kernel.

    Args:
        size (int): The size of the kernel. Should be an odd number.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A 2D Gaussian kernel of shape (size, size).
    """
    # Ensure size is an odd number
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Create a coordinate grid of size x size
    ax = torch.arange(-size // 2 + 1, size // 2 + 1).float()
    xx, yy = torch.meshgrid(ax, ax)
    
    # Calculate the 2D Gaussian kernel
    kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    
    # Normalize the kernel to ensure the sum is 1
    kernel = kernel / torch.sum(kernel)
    
    return kernel

class MixtureSirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model. If it is, no
            activation is applied and 0.5 is added to the output. Since we
            assume all training data lies in [0, 1], this allows for centering
            the output of the model.
        use_bias (bool): Whether to learn bias in linear layer.
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        k_mixture,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
        siren_init=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.use_bias = use_bias
        self.mixture_weight = nn.Parameter(torch.ones(k_mixture, dim_in, dim_out))
        if use_bias:
            self.mixture_bias = nn.Parameter(torch.zeros(k_mixture, dim_out))

        # Initialize layers following SIREN paper
        # print(siren_init)
        # import pdb; pdb.set_trace()
        print(f'w0 {w0}')
        if siren_init:
            print(f'siren init')    
            w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
            nn.init.uniform_(self.mixture_weight, -w_std, w_std)
            if use_bias:
                nn.init.uniform_(self.mixture_bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x, coeff):
        '''
        x: (B, N, dim_in)
        weight: (B, dim_in, dim_out)
        bias: (B, dim_out)
        coeff: (B, k)
        '''
        weight = torch.sum(coeff.unsqueeze(-1).unsqueeze(-1) * self.mixture_weight, dim=1)
        out = torch.matmul(x, weight)
        if self.use_bias:
            bias = torch.sum(coeff.unsqueeze(-1) * self.mixture_bias, dim=1, keepdim=True)
            out = out + bias
        # out = self.linear(x)
        if self.is_last:
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            out += 0
        else:
            out = self.activation(out)
        return out

class ModulatedSirenMoW(nn.Module):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        modulate_scale (bool): Whether to modulate with scales.
        modulate_shift (bool): Whether to modulate with shifts.
        use_latent (bool): If true, use a latent vector which is mapped to
            modulations, otherwise use modulations directly.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Number of hidden dimensions of
            modulation network.
        modulation_net_num_layers (int): Number of layers in modulation network.
            If this is set to 1 will correspond to a linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        modulate_scale=False,
        modulate_shift=True,
        use_latent=False,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_layers=1,
        mu=0,
        sigma=1,
        last_activation=None,
        siren_init=True,
        k_mixture=256,
    ):
        super().__init__()
        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                MixtureSirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    k_mixture=k_mixture,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    siren_init=siren_init
                )
            )
        self.net = nn.ModuleList(layers)
        self.last_layer = MixtureSirenLayer(
            dim_in=dim_hidden, dim_out=dim_out, k_mixture=k_mixture, w0=w0, use_bias=use_bias, is_last=True, siren_init=siren_init
        )

        # Must modulate at least one of scale and shift
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
    
        self.mu = mu
        self.sigma = sigma
        self.last_activation = (
            nn.Identity() if last_activation is None else last_activation
        )
       
        num_modulations = k_mixture * (num_layers)
        self.num_layers = num_layers
        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
            nn.init.constant_(self.modulation_net.net.bias, 1.0/k_mixture)
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations
        
    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        # import pdb; pdb.set_trace()
        mixture_coeff = self.modulation_net(latent).reshape(x.shape[0], self.num_layers, -1)
        # idx = 0
        for idx, module in enumerate(self.net):
        
            x = module(x, mixture_coeff[:, idx])
            # idx = idx + self.dim_hidden

        # Shape (batch_size, num_points, dim_out)
        out = self.last_activation(self.last_layer(x, mixture_coeff[:,-1]))
        out = out * self.sigma + self.mu
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])