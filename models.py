import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from collections import OrderedDict
import numpy as np
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Callable,
    Optional,
    Generator,
    Collection,
)
ListRules = Collection[Callable[[str], str]]
LRScheduler = _LRScheduler
ModelParams = Generator[Tensor, Tensor, Tensor]
NormLayers = Union[torch.nn.Identity, torch.nn.LayerNorm, torch.nn.BatchNorm1d]
allowed_activations = [
    "relu",
    "leaky_relu",
    "tanh",
    "gelu",
    "geglu",
    "reglu",
    "softplus",]


    
class FlowEncoder(nn.Module):
    def __init__(self, 
                 column_idx, 
                 embed_input,
                 continuous_cols, 
                 dropout=0.3, 
                 num_classes=8, 
                 projection_dim=64,
                 ):

        super(FlowEncoder, self).__init__()
        self.column_idx = column_idx
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.dropout = dropout
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.encoder, self.features_dim = self._create_encoder()
        
        # head
        self.projection_head = nn.Sequential(
                # nn.Linear(self.features_dim, self.features_dim),
                # nn.GELU(),
                nn.Linear(self.features_dim, self.projection_dim))
        
        # cls
        self.cls = nn.Sequential(
                nn.Linear(self.features_dim, self.num_classes))
        
    def projection(self, x):
        out = self.projection_head(x)
        out = F.normalize(out, dim=1)
        return out
    
    def classifier(self, x):
        out = self.cls(x)
        return out
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.cls(out)
        return out
    
    
    
    

    def _create_encoder(self):
        model = TabResnet(
            blocks_dims=[128, 256, 512],
            column_idx=self.column_idx,
            embed_input=self.embed_input,
            continuous_cols=self.continuous_cols,
            cont_norm_layer="layernorm",
            concat_cont_first=True,
            embed_dropout=self.dropout, 
            blocks_dropout=self.dropout,
            activation='gelu',
            mlp_hidden_dims=[512],
            mlp_dropout=self.dropout,   
            mlp_activation="gelu",          
            mlp_linear_first=True,
            mlp_batchnorm=False, 
            mlp_batchnorm_last=False, 
            )

        for param in model.parameters():
            pass
        fea_dim = param.shape[0]
        return model, fea_dim



###########################################################
# source code: https://github.com/jrzaurin/pytorch-widedeep 
# @author: jrzaurin(https://github.com/jrzaurin)
###########################################################

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class REGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if activation == "tanh":
        return nn.Tanh()
    if activation == "gelu":
        return nn.GELU()
    if activation == "geglu":
        return GEGLU()
    if activation == "reglu":
        return REGLU()
    if activation == "softplus":
        return nn.Softplus()

def dense_layer(
    inp: int,
    out: int,
    activation: str,
    p: float,
    bn: bool,
    linear_first: bool,):
    # This is basically the LinBnDrop class at the fastai library
    if activation == "geglu":
        raise ValueError(
            "'geglu' activation is only used as 'transformer_activation' "
            "in transformer-based models"
        )
    act_fn = get_activation_fn(activation)
    layers = [nn.BatchNorm1d(out if linear_first else inp)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))  # type: ignore[arg-type]
    lin = [nn.Linear(inp, out, bias=not bn), act_fn]
    layers = lin + layers if linear_first else layers + lin
    return nn.Sequential(*layers)

class CatEmbeddingsAndCont(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        embed_dropout: float,
        continuous_cols: Optional[List[str]],
        cont_norm_layer: str,
    ):
        super(CatEmbeddingsAndCont, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols

        # Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict(
                {
                    "emb_layer_" + col: nn.Embedding(val + 1, dim, padding_idx=0)
                    for col, val, dim in self.embed_input
                }
            )
            self.embedding_dropout = nn.Dropout(embed_dropout)
            self.emb_out_dim: int = int(
                np.sum([embed[2] for embed in self.embed_input])
            )
        else:
            self.emb_out_dim = 0

        # Continuous
        if self.continuous_cols is not None:
            self.cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            self.cont_out_dim: int = len(self.continuous_cols)
            if cont_norm_layer == "batchnorm":
                self.cont_norm: NormLayers = nn.BatchNorm1d(self.cont_out_dim)
            elif cont_norm_layer == "layernorm":
                self.cont_norm = nn.LayerNorm(self.cont_out_dim)
            else:
                self.cont_norm = nn.Identity()
        else:
            self.cont_out_dim = 0

        self.output_dim = self.emb_out_dim + self.cont_out_dim

    def forward(self, X: Tensor) -> Tuple[Tensor, Any]:
        if self.embed_input is not None:
            embed = [
                self.embed_layers["emb_layer_" + col](X[:, self.column_idx[col]].long())
                for col, _, _ in self.embed_input
            ]
            x_emb = torch.cat(embed, 1)
            x_emb = self.embedding_dropout(x_emb)
        else:
            x_emb = None
        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
        else:
            x_cont = None

        return x_emb, x_cont

class MLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        activation: str,
        dropout: Optional[Union[float, List[float]]],
        batchnorm: bool,
        batchnorm_last: bool,
        linear_first: bool,
    ):
        super(MLP, self).__init__()

        if not dropout:
            dropout = [0.0] * len(d_hidden)
        elif isinstance(dropout, float):
            dropout = [dropout] * len(d_hidden)

        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module(
                "dense_layer_{}".format(i - 1),
                dense_layer(
                    d_hidden[i - 1],
                    d_hidden[i],
                    activation,
                    dropout[i - 1],
                    batchnorm and (i != len(d_hidden) - 1 or batchnorm_last),
                    linear_first,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)

class TabMlp(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: Optional[List[Tuple[str, int, int]]] = None,
        embed_dropout: float = 0.1,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: str = "batchnorm",
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "gelu",
        mlp_dropout: Union[float, List[float]] = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = False,
        
    ):
        super(TabMlp, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.mlp_hidden_dims = mlp_hidden_dims
        self.embed_dropout = embed_dropout
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_linear_first = mlp_linear_first
        

        if self.mlp_activation not in allowed_activations:
            raise ValueError(
                "Currently, only the following activation functions are supported "
                "for for the MLP's dense layers: {}. Got {} instead".format(
                    ", ".join(allowed_activations), self.mlp_activation
                )
            )

        self.cat_embed_and_cont = CatEmbeddingsAndCont(
            column_idx,
            embed_input,
            embed_dropout,
            continuous_cols,
            cont_norm_layer,
        )

        # MLP
        mlp_input_dim = self.cat_embed_and_cont.output_dim
        mlp_hidden_dims = [mlp_input_dim] + mlp_hidden_dims
        self.tab_mlp = MLP(
            mlp_hidden_dims,
            mlp_activation,
            mlp_dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = mlp_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:
        r"""Forward pass that concatenates the continuous features with the
        embeddings. The result is then passed through a series of dense layers
        """
        x_emb, x_cont = self.cat_embed_and_cont(X)
        if x_emb is not None:
            x = x_emb
        if x_cont is not None:
            x = torch.cat([x, x_cont], 1) if x_emb is not None else x_cont
        return self.tab_mlp(x)

class BasicBlock(nn.Module):
    def __init__(self, inp: int, out: int, dropout: float = 0.0, resize: Module = None, activation: str = 'gelu'):
        super(BasicBlock, self).__init__()

        self.lin1 = nn.Linear(inp, out)
        #self.bn1 = nn.BatchNorm1d(out)
        self.ln1 =nn.LayerNorm(out)
        #self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.act_fn = get_activation_fn(activation)
        if dropout > 0.0:
            self.dropout = True
            self.dp = nn.Dropout(dropout)
        else:
            self.dropout = False
        self.lin2 = nn.Linear(out, out)
        #self.bn2 = nn.BatchNorm1d(out)
        self.ln2 = nn.LayerNorm(out)
        self.resize = resize

    def forward(self, x):

        identity = x

        out = self.lin1(x)
        out = self.ln1(out)
        out = self.act_fn(out)
        if self.dropout:
            out = self.dp(out)

        out = self.lin2(out)
        out = self.ln2(out)

        if self.resize is not None:
            identity = self.resize(x)

        out += identity
        out = self.act_fn(out)

        return out

class DenseResnet(nn.Module):
    def __init__(self, input_dim: int, blocks_dims: List[int], dropout: float, activation: str):
        super(DenseResnet, self).__init__()

        self.input_dim = input_dim
        self.blocks_dims = blocks_dims
        self.dropout = dropout
        self.activation = activation  

        if input_dim != blocks_dims[0]:
            self.dense_resnet = nn.Sequential(
                OrderedDict(
                    [
                        ("lin1", nn.Linear(input_dim, blocks_dims[0]))#,
                        #("bn1", nn.BatchNorm1d(blocks_dims[0]))
                        #nn.LayerNorm(self.cont_out_dim)
                        #("ln1", nn.LayerNorm(blocks_dims[0]))
                    ]
                )
            )
        else:
            self.dense_resnet = nn.Sequential()
        for i in range(1, len(blocks_dims)):
            resize = None
            if blocks_dims[i - 1] != blocks_dims[i]:
                resize = nn.Sequential(
                    nn.Linear(blocks_dims[i - 1], blocks_dims[i]),
                    # nn.BatchNorm1d(blocks_dims[i]
                    nn.LayerNorm(blocks_dims[i]))
                
            self.dense_resnet.add_module(
                "block_{}".format(i - 1),
                BasicBlock(blocks_dims[i - 1], blocks_dims[i], dropout, resize, self.activation)
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.dense_resnet(X)

class TabResnet(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: Optional[List[Tuple[str, int, int]]] = None,
        embed_dropout: float = 0.0,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: str = "layernorm",
        concat_cont_first: bool = True,
        blocks_dims: List[int] = [256, 256, 256],
        blocks_dropout: float = 0.0,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "gelu",
        mlp_dropout: float = 0.0,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = False,
        activation: str = 'gelu'
    ):
        super(TabResnet, self).__init__()

        if len(blocks_dims) < 2:
            raise ValueError(
                "'blocks' must contain at least two elements, e.g. [256, 128]"
            )

        if not concat_cont_first and embed_input is None:
            raise ValueError(
                "If 'concat_cont_first = False' 'embed_input' must be not 'None'"
            )

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.embed_dropout = embed_dropout
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.concat_cont_first = concat_cont_first
        self.blocks_dims = blocks_dims
        self.blocks_dropout = blocks_dropout
        self.mlp_activation = mlp_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.cat_embed_and_cont = CatEmbeddingsAndCont(
            column_idx,
            embed_input,
            embed_dropout,
            continuous_cols,
            cont_norm_layer,
        )

        emb_out_dim = self.cat_embed_and_cont.emb_out_dim
        cont_out_dim = self.cat_embed_and_cont.cont_out_dim

        # DenseResnet
        if self.concat_cont_first:
            dense_resnet_input_dim = emb_out_dim + cont_out_dim
            self.output_dim = blocks_dims[-1]
        else:
            dense_resnet_input_dim = emb_out_dim
            self.output_dim = cont_out_dim + blocks_dims[-1]
        self.tab_resnet_blks = DenseResnet(
            dense_resnet_input_dim, blocks_dims, blocks_dropout, activation
        )


        # MLP
        if self.mlp_hidden_dims is not None:
            if self.concat_cont_first:
                mlp_input_dim = blocks_dims[-1]
            else:
                mlp_input_dim = cont_out_dim + blocks_dims[-1]
            mlp_hidden_dims = [mlp_input_dim] + mlp_hidden_dims
            self.tab_resnet_mlp = MLP(
                mlp_hidden_dims,
                mlp_activation,
                mlp_dropout,
                mlp_batchnorm,
                mlp_batchnorm_last,
                mlp_linear_first,
            )
            self.output_dim = mlp_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:
        r"""Forward pass that concatenates the continuous features with the
        embeddings. The result is then passed through a series of dense Resnet
        blocks"""

        x_emb, x_cont = self.cat_embed_and_cont(X)

        if x_cont is not None:
            if self.concat_cont_first:
                x = torch.cat([x_emb, x_cont], 1) if x_emb is not None else x_cont
                out = self.tab_resnet_blks(x)
                
            else:
                out = torch.cat([self.tab_resnet_blks(x_emb), x_cont], 1)
        else:
            out = self.tab_resnet_blks(x_emb)
        
        if self.mlp_hidden_dims is not None:
            out = self.tab_resnet_mlp(out)
   
        return out
