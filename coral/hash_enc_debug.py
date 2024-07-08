import tinycudann as tcnn 
import torch 

enc = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 2,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 8,
                "per_level_scale": 1.5,
            },
        )

x = torch.rand(1,5,2).repeat(2,1,1)
y = enc(x.reshape(-1,2))
y = y.reshape(2,5,4)
assert torch.all(y[0]==y[1])
print(y.shape)