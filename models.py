import torch
from torch import nn
from torch.nn import init

# JK
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform(m.weight.data, a=0, mode='fan_in')
                init.kaiming_uniform(m.bias.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
                init.orthogonal_(m.bias.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



class ScoreModel(nn.Module):
    def __init__(self, clamp=False, masked_feat_id=None):
        super(ScoreModel, self).__init__()
        self.clamp = clamp
        self.masked_feat_id = masked_feat_id

    def compute_score(self, x):
        raise NotImplementedError

    def forward(self, x):
        if self.masked_feat_id is not None:
            x = x.index_fill(-1, torch.tensor(self.masked_feat_id, dtype=torch.long, device=x.device), 0.0)
        score = self.compute_score(x)
        if self.clamp:
            score = torch.clamp(score, -10, 10)
        return score


class LinearModel(ScoreModel):

    #One layer simple linear model


    def __init__(self, input_dim=2, **kwargs):
        super(LinearModel, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.w = nn.Linear(input_dim, 1, bias=True)

    def compute_score(self, x):
        h = self.w(x)
        return h


"""   original
class MLP(ScoreModel):
    def __init__(self, input_dim: int, hidden_layer: int, dropout: float = 0.0, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.input_size = input_dim
        fcs = []
        last_size = self.input_size
        for _ in range(hidden_layer):
            size = last_size // 2
            linear = nn.Linear(last_size, size)
            linear.bias.data.fill_(0.0)
            fcs.append(linear)
            last_size = size
            fcs.append(nn.ReLU())
            if dropout > 0.0:
                fcs.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fcs)
        linear = nn.Linear(last_size, 1)
        linear.bias.data.fill_(0.0)
        self.final_layer = linear

    def compute_score(self, x):
        out = self.fc(x)
        out = self.final_layer(out)
        return out
"""

# Batch Norm version
class MLP(ScoreModel):
    def __init__(self, input_dim: int, hidden_layer: int, dropout: float = 0.0, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.input_size = input_dim
        fcs = []
        last_size = self.input_size
        for _ in range(hidden_layer):
            size = last_size // 2
            linear = nn.Linear(last_size, size)
            bn = nn.BatchNorm1d(20)  #JK


            #bn = nn.LayerNorm(size)  #JK
            linear.bias.data.fill_(0.0)
            fcs.append(linear)
            use_bn = False #JK
            if use_bn:    #JK
                fcs.append(bn) #JK
            last_size = size
            fcs.append(nn.ReLU())
            if dropout > 0.0:
                fcs.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fcs)
        linear = nn.Linear(last_size, 1)
        linear.bias.data.fill_(0.0)
        self.final_layer = linear

    def compute_score(self, x):

        out = self.fc(x)
        out = self.final_layer(out)
        return out

# MLP for predicting an embedding of the group composition.
# Differs from the main MLP in that each item does not get its own score
# Items are aggregated to a group composition embedding to help predict accurate
class MLPGroupEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_layer: int, dropout: float = 0.0, clamp = False):
        super(MLPGroupEmbedding, self).__init__()
        self.input_size = input_dim
        self.clamp = clamp
        fcs = []
        input_size = self.input_size
        for _ in range(hidden_layer):
            #size = last_size // 2
            #linear = nn.Linear(last_size, size)
            linear = nn.Linear(input_size, input_size)
            linear.bias.data.fill_(0.0)
            fcs.append(linear)
            bn = nn.BatchNorm1d(20)    # JK 1005
            #last_size = size
            fcs.append( bn )           # JK 1005
            fcs.append(nn.ReLU())
            if dropout > 0.0:
                fcs.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fcs)
        linear = nn.Linear(input_size, input_size)
        linear.bias.data.fill_(0.0)
        self.final_layer = linear

    def forward(self, x):
        out = self.fc(x)
        out = self.final_layer(out)
        if self.clamp:
            out = torch.clamp(out, -10, 10)
        return out


# JK
class SiameseMLP(nn.Module):
    def __init__(self, ScoreModel, GroupCompModel):
        super(SiameseMLP, self).__init__()
        self.gcm = GroupCompModel
        self.sm  = ScoreModel

    def forward(self,x,gid):

        out_sm  = self.sm(x)
        out_gcm = self.gcm(gid)

        return out_sm, out_gcm


# This version concatenates item scores with group id vector, and
# aggregates to size-N^2 score embedding for the QP
class MLPQuadScore(nn.Module):
    def __init__(self, ScoreModel, list_len):
        super(MLPQuadScore, self).__init__()
        self.sm  = ScoreModel
        self.fc1 = nn.Linear(2*list_len,list_len**2)
        self.fc2 = nn.Linear(list_len**2,list_len**2)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)
        self.bn  = nn.BatchNorm1d(400)

    def forward(self,x,gid):

        out_sm  = self.sm(x)

        #out = torch.cat( (out_sm.squeeze(-1),gid),1 )
        out = torch.cat( (out_sm.view(gid.shape),gid),1 )
        out = self.fc1(out)
        out = self.bn(out)
        out = self.fc2(out)

        return out



def convert_vars_to_gpu(varlist):
    return [var.cuda() for var in varlist]






# for Zehlike baseline 11/8

class NNModel(nn.Module):
    """
    Neural network model
    """

    def __init__(self,
                 hidden_layer=64,
                 D=2,
                 dropout=0.0,
                 init_weight1=None,
                 init_weight2=None,
                 pooling=False,
                 clamp=False):
        self.input_dim = D
        super(NNModel, self).__init__()
        self.fc = nn.Linear(D, hidden_layer, bias=True)
        self.fc_drop = nn.Dropout(p=dropout)
        # self.activation = nn.ReLU()
        self.activation = nn.ReLU()
        if pooling == "concat_avg":
            self.fc2 = nn.Linear(2 * hidden_layer, hidden_layer, bias=True)
            self.fc3 = nn.Linear(hidden_layer, 1, bias=True)
        elif pooling is not False:
            self.fc2 = nn.Linear(hidden_layer, hidden_layer, bias=True)
            self.fc3 = nn.Linear(hidden_layer, 1, bias=True)
        else:
            self.fc2 = nn.Linear(hidden_layer, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        if init_weight1 is not None:
            self.fc.weight = torch.nn.Parameter(init_weight1)
        if init_weight2 is not None:
            self.fc2.weight = torch.nn.Parameter(init_weight2)
        self.pooling_layer = pooling
        self.clamp = clamp

    def forward(self, x):
        h = self.activation(self.fc(x))
        h = self.fc_drop(h)
        if self.pooling_layer:
            if self.pooling_layer == "average":
                h1 = h - torch.mean(h, dim=0, keepdim=True)
            elif self.pooling_layer == "max":
                h1 = h - torch.max(h, dim=0, keepdim=True)
            elif self.pooling_layer == "concat_avg":
                h1 = torch.cat(
                    (h, torch.mean(h, dim=0, keepdim=True).repeat(
                        x.size()[0], 1)),
                    dim=1)
        else:
            h1 = h
        h2 = self.fc2(h1)
        if self.pooling_layer:
            h3 = self.fc3(self.activation(h2))
            return h3 if not self.clamp else torch.clamp(h3, -10, 10)

        else:
            return h2 if not self.clamp else torch.clamp(h2, -10, 10)
