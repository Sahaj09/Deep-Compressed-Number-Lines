import torch
import torch.nn as nn

from WM import WM



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.ones_(m.weight)
        m.bias.data.fill_(0)


class alpha_prediction(nn.Module):
    # predicts alpha value for each feature in the input stream

    def __init__(self, n_extern, n_inputs):
        super().__init__()
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(n_extern, int(n_extern/2)), #, bias=False),
            nn.ReLU(),
            nn.Linear(int(n_extern/2) ,n_inputs),
            nn.ReLU()
        )
        self.fully_connected_layers.apply(init_weights) # check and change for a more complex structure.

    def forward(self, input):
        return self.fully_connected_layers(input)


class alpha_prediction_attention(nn.Module):
    # predicts alpha value for each feature in the input stream

    def __init__(self, n_extern, n_inputs):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(n_inputs)

    def forward(self, input):
        return self.fully_connected_layers(input)



class f_tilda_prediction(nn.Module):
    # predicts label from f_tilda

    def __init__(self, inputs, outputs):
        super().__init__()

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(inputs, outputs)
            )

    def forward(self, input):
        return self.fully_connected_layers(input)


class dim_reduce_network(nn.Module):
    # reduces input feature dimensions for input to SITH if required.

    def __init__(self, inputs, outputs):
        super().__init__()

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(inputs, outputs),
            nn.ReLU()
            )

    def forward(self, input):
        return self.fully_connected_layers(input)




class WMPred(nn.Module):

    def __init__(self, sith, n_inputs, n_outputs, n_extern = None,  high_dim_input = None, use_F = False):
        super().__init__()
        """
        sith : instance of the  WM class
        n_inputs : size of the input to sith (or) number of features given as input to sith
        n_outputs : size of the output/prediction
        n_extern : size of the external stimulus. If set to None, no external stimulus exists.
        high_dim_input : If using high dimensional input, projects the input features from size high_dim_input to n_inputs using a dense net.
        use_F : if True, we don't calculate f_tilda using post inversion
        """
        self.sith = sith
        self.n_inputs = n_inputs
        self.n_extern = n_extern
        self.high_dim_input = high_dim_input
        self.use_F = use_F

        if n_extern == 0:
            self.n_extern = None
        if high_dim_input == 0:
            self.high_dim_input = None

        
        self.alpha = None
        self.til_f = None
        self.F = None
        self.p = None
        self.f = None
        self.flatten = nn.Flatten(start_dim=2)

        #print("check n_extern",n_extern)
        
        if self.n_extern!=None:
            self.alpha_pred = alpha_prediction(n_extern, n_inputs)

        if self.high_dim_input!=None:
            self.dim_reduce = dim_reduce_network(high_dim_input, n_inputs)

        self.f_tilda_pred = f_tilda_prediction(n_inputs * sith.n_taus, n_outputs)

        
        
        

    def forward(self, x, h):

        # selecting input to sith
        if self.high_dim_input==None:
            self.f = x[..., :self.n_inputs]
        else:
            f = x[..., :self.high_dim_input]
            self.f= self.dim_reduce(f)

        # selecting alpha for each feature
        if self.n_extern!=None:
            z = x[..., -self.n_extern:]
            self.alpha = self.alpha_pred(z)
        else:
            self.alpha = None


        # passing selected input and alpha to sith and calculating f_tilda
        self.til_f, h, self.F = self.sith(self.f, h, alpha= self.alpha, delta=None)

        # calculating prediction from f_tilda or F.
        if self.use_F==False:
            self.p = self.f_tilda_pred(self.flatten(self.til_f))
        else:
            self.p = self.f_tilda_pred(self.flatten(self.F))

        return self.p, h



class WMPred_with_bptt(nn.Module):
    def __init__(self, sith, n_inputs, n_outputs, n_extern = None,  high_dim_input = None, use_F = False):
        super().__init__()

        self.model = WMPred(sith, n_inputs, n_outputs, n_extern,  high_dim_input, use_F)

    def forward(self, x, h):
        # x - shape should be batch, seq_len, features
        total_seq_length = x.size()[1]

        for i in range(0,total_seq_length):
            o, h = self.model(x[:,i:i+1,:], h)
            
            if i==0:
                output = o
            else:
                output = torch.cat((output,o), 1)

        #print("check single output - ", o.size())
        #print("check all outputs combined- ", output.size())

        return output, h



