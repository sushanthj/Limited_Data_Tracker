from torch import nn
import torch
import numpy as np

class LayerNorm(nn.Module):
    """
    Using only channel_last method which is implemented in torch's layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-6, channels="first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
        self.channels = channels

    def forward(self, x):
        if self.channels == "last":
            """
            To use inbuilt layer_norm we permute from
            (batch_size, channels, height, width) -> (batch_size, height, width, channels)
            """
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.channels == "first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DropPath(nn.Module):
    """
    Stochastic Depth (we drop the non-shortcut path inside residual blocks with
                      some probability p)
    """

    def __init__(self, drop_probability = 0.0):
        super().__init__()
        self.drop_prob = drop_probability

    def forward(self, x):
        # if drop prob is zero or in inference mode, skip this
        if np.isclose(self.drop_prob, 0.0, atol=1e-9) or not self.training:
          return x

        # find output shape (eg. if input = 4D tensor, output = (1,1,1,1))
        # output_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        output_shape = (x.shape[0],1,1,1)

        # create mask of output shape and of input type on same device
        keep_mask = torch.empty(output_shape, dtype=x.dtype, device=x.device).bernoulli_((1-self.drop_prob))
        # Alternative: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        # NOTE: all methods like bernoulli_ with the underscore suffix means they
        # are inplace operations
        # The div_ function is used here to normalize the keep_mask tensor.
        # This is done to ensure that the expected sum of the tensor remains
        # the same before and after applying dropout, which is important for
        # maintaining the expected activation size.
        keep_mask.div_((1-self.drop_prob))

        return x*keep_mask

class ConvNextBlock(nn.Module):
    """
    Refer : https://browse.arxiv.org/pdf/2201.03545v2.pdf for detailed architechture

    """

    def __init__(self, num_ch, expansion_factor, drop_prob=0.0, config={}):
        # num_ch = number of channels at first and third layer of block
        # There'll be an expansion in the second layer given by expansion_factor
        super().__init__()

        """
        NOTE: To perform depthwise conv we use the param (groups=num_ch)
        to create a separate filter for each input channel
        """


        self.main_block = nn.Sequential(
            # 1st conv layer (deptwise)
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch,
                            kernel_size=7, padding=3, groups=num_ch),
            nn.BatchNorm2d(num_ch),

            # 2nd conv layer
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch*expansion_factor, kernel_size=1, stride=1), # 1x1 pointwise convs implemented as Linear Layer
            nn.GELU(),

            # 3rd conv layer
            nn.Conv2d(in_channels=num_ch*expansion_factor, out_channels=num_ch, kernel_size=1, stride=1)
        )

        for layer in self.main_block:
            if isinstance(layer, nn.Conv2d):
                nn.init.trunc_normal_(layer.weight,
                                      mean=config.get('truncated_normal_mean', 0),
                                      std=config.get('truncated_normal_std', 0.2))
                nn.init.constant_(layer.bias, 0)

        # define the drop_path layer
        if drop_prob > 0.0:
            self.drop_residual_path = DropPath(drop_prob)
        else:
            self.drop_residual_path = nn.Identity()

    def forward(self, x):
        input = x.clone()
        x = self.main_block(x)

        # sum the main and shortcut connection
        x = input + self.drop_residual_path(x)

        return x


class Network(torch.nn.Module):
    """
    ConvNext
    """

    def __init__(self, num_classes=1, drop_rate=0.5, expand_factor=4, config={}):
        super().__init__()

        self.backbone_out_channels = 120
        self.num_classes = num_classes

        # number of channels at input/output of each res_blocks
        self.channel_list = [30, 60, 80, 120]
        # self.channel_list = [96, 192, 384, 768]

        # number of repeats for each res_block
        self.block_repeat_list = [2,2,3,2]
        # self.block_repeat_list = [3,3,9,3]

        # define number of stages from above
        self.num_stages = len(self.block_repeat_list)

        self.drop_path_probabilities = [i.item() for i in torch.linspace(0, drop_rate, sum(self.channel_list))]

        ############## DEFINE RES BLOCK AND AUX LAYERS ########################

        # # Define the Stem (the first layer which takes input images)
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=self.channel_list[0], kernel_size=4, stride=4),
            torch.nn.BatchNorm2d(self.channel_list[0]),
            )

        # truncated normal initialization
        for layer in self.stem:
            if isinstance(layer, torch.nn.Conv2d):
                nn.init.trunc_normal_(layer.weight,
                                      mean=config.get('truncated_normal_mean', 0),
                                      std=config.get('truncated_normal_std', 0.2))
                nn.init.constant_(layer.bias, 0)

        # # Store the LayerNorm and Downsampling layer when switching btw 2 types of res_blocks
        # self.block_to_block_ln_and_downsample = []
        self.block_to_block_ln_and_downsample = [self.stem]
        for i in range(self.num_stages - 1):
            inter_downsample = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(self.channel_list[i]),
                    torch.nn.Conv2d(in_channels=self.channel_list[i],
                                    out_channels=self.channel_list[i+1],
                                    kernel_size=2, stride=2)
                  )
            self.block_to_block_ln_and_downsample.append(inter_downsample)

        # Store the Res_block stages (eg. 3xres_2, 3xres_3, ...)
        self.res_block_stages = torch.nn.ModuleList()
        for i in range(self.num_stages):
            res_block_layer = []
            for j in range(self.block_repeat_list[i]):
                res_block_layer.append(ConvNextBlock(num_ch=self.channel_list[i],
                                  expansion_factor=expand_factor,
                                  drop_prob=self.drop_path_probabilities[i+j], config=config))

            # append the repeated res_blocks as one layer
            # *res_block_layer means we add individual elements of the res_block_layer list
            self.res_block_stages.append(torch.nn.Sequential(*res_block_layer))

        # truncated normal initialization
        for res_block_stage in self.res_block_stages:
            for layer in res_block_stage:
                if isinstance(layer, torch.nn.Conv2d):
                    nn.init.trunc_normal_(layer.weight,
                                          mean=config.get('truncated_normal_mean', 0),
                                          std=config.get('truncated_normal_std', 0.2))
                    nn.init.constant_(layer.bias, 0)

        # # final norm layer (here we use torch's own)
        # self.final_norm = torch.nn.LayerNorm(self.channel_list[-1], eps=1e-6)

        # # final pool layer
        # self.final_pool = torch.nn.AdaptiveAvgPool2d((1,1))

        #####################################################################

        self.backbone = torch.nn.Sequential(
              # essentially stem (replace with stem if it works)
              self.block_to_block_ln_and_downsample[0],
              # res_1 block
              self.res_block_stages[0],
              self.block_to_block_ln_and_downsample[1],
              # res_2 block
              self.res_block_stages[1],
              self.block_to_block_ln_and_downsample[2],
              # res_3 block
              self.res_block_stages[2],
              self.block_to_block_ln_and_downsample[3],
              # res_4 block
              self.res_block_stages[3],
              torch.nn.AdaptiveAvgPool2d((1,1)),
              torch.nn.Flatten(),
            )

        self.cls_layer = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_out_channels, self.num_classes))

        # truncated normal initialization
        for layer in self.cls_layer:
            if isinstance(layer, torch.nn.Linear):
                nn.init.trunc_normal_(layer.weight, mean=config['truncated_normal_mean'], std=config['truncated_normal_std'])
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, return_feats=False):
        """
        What is return_feats? It essentially returns the second-to-last-layer
        features of a given image. It's a "feature encoding" of the input image,
        and you can use it for the verification task. You would use the outputs
        of the final classification layer for the classification task.

        You might also find that the classification outputs are sometimes better
        for verification too - try both.
        """
        feats = self.backbone(x)
        feats = nn.functional.dropout(feats, p=0.5, training=self.training)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out