import torch
import torch.nn as nn

#----for LMU
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
import numpy as np
from functools import partial
import torch.nn.functional as F
import math
from nengolib.signal import Identity, cont2discrete # comment this to have everything working without LMU in puthon versions > 3.6, LMU is not supported beyond that
from nengolib.synapses import LegendreDelay   # comment this to have everything working without LMU in puthon versions > 3.6, LMU is not supported beyond that
from functools import partial
#----

#----for CoRNN
from torch import nn
import torch
from torch.autograd import Variable
#----

class RNN1FC(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_rnn_hidden):
        super().__init__()

        self.rnn = nn.RNN(n_inputs, n_rnn_hidden, batch_first=True)
        self.fc = nn.Linear(n_rnn_hidden, n_outputs)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        rnn_output, h = self.rnn(x, h)
        p = self.fc(rnn_output)
        return p, h


class RNN2FC(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_rnn_hidden, n_fc_hidden):
        super().__init__()

        self.rnn = nn.RNN(n_inputs, n_rnn_hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(n_rnn_hidden, n_fc_hidden),
            nn.ReLU(),
            nn.Linear(n_fc_hidden, n_outputs)
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        rnn_output, h = self.rnn(x, h)
        p = self.fc(rnn_output)
        return p, h


class LSTM1FC(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_rnn_hidden):
        super().__init__()

        self.lstm = nn.LSTM(n_inputs, n_rnn_hidden, batch_first=True)
        self.fc = nn.Linear(n_rnn_hidden, n_outputs)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        lstm_output, h = self.lstm(x, h)
        p = self.fc(lstm_output)
        return p, h


class LSTM2FC(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_rnn_hidden, n_fc_hidden):
        super().__init__()

        self.lstm = nn.LSTM(n_inputs, n_rnn_hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(n_rnn_hidden, n_fc_hidden),
            nn.ReLU(),
            nn.Linear(n_fc_hidden, n_outputs)
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        lstm_output, h = self.lstm(x, h)
        p = self.fc(lstm_output)
        return p, h


class GRU1FC(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_rnn_hidden):
        super().__init__()

        self.gru = nn.GRU(n_inputs, n_rnn_hidden, batch_first=True)
        self.fc = nn.Linear(n_rnn_hidden, n_outputs)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        gru_output, h = self.gru(x, h)
        p = self.fc(gru_output)
        return p, h


class GRU2FC(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_rnn_hidden, n_fc_hidden):
        super().__init__()

        self.gru = nn.GRU(n_inputs, n_rnn_hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(n_rnn_hidden, n_fc_hidden),
            nn.ReLU(),
            nn.Linear(n_fc_hidden, n_outputs)
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        gru_output, h = self.gru(x, h)
        p = self.fc(gru_output)
        return p, h



#------------------------------------ LMU ----------------------------- https://github.com/AbdouJaouhar/LMU-Legendre-Memory-Unit



def lecun_uniform(tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))


class LMUCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                 order,
                 theta=100,  # relative to dt=1
                 method='zoh',
                 trainable_input_encoders=True,
                 trainable_hidden_encoders=True,
                 trainable_memory_encoders=True,
                 trainable_input_kernel=True,
                 trainable_hidden_kernel=True,
                 trainable_memory_kernel=True,
                 trainable_A=False,
                 trainable_B=False,
                 input_encoders_initializer=lecun_uniform,
                 hidden_encoders_initializer=lecun_uniform,
                 memory_encoders_initializer=partial(torch.nn.init.constant_, val=0),
                 input_kernel_initializer=torch.nn.init.xavier_normal_,
                 hidden_kernel_initializer=torch.nn.init.xavier_normal_,
                 memory_kernel_initializer=torch.nn.init.xavier_normal_,

                 hidden_activation='tanh',
                 ):
        super(LMUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.order = order

        if hidden_activation == 'tanh':
            self.hidden_activation = torch.tanh
        elif hidden_activation == 'relu':
            self.hidden_activation = torch.relu
        else:
            raise NotImplementedError("hidden activation '{}' is not implemented".format(hidden_activation))

        realizer = Identity()
        self._realizer_result = realizer(
            LegendreDelay(theta=theta, order=self.order))
        self._ss = cont2discrete(
            self._realizer_result.realization, dt=1., method=method)
        self._A = self._ss.A - np.eye(order)  # puts into form: x += Ax
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        self.input_encoders = nn.Parameter(torch.Tensor(1, input_size), requires_grad=trainable_input_encoders)
        self.hidden_encoders = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=trainable_hidden_encoders)
        self.memory_encoders = nn.Parameter(torch.Tensor(1, order), requires_grad=trainable_memory_encoders)
        self.input_kernel = nn.Parameter(torch.Tensor(hidden_size, input_size), requires_grad=trainable_input_kernel)
        self.hidden_kernel = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=trainable_hidden_kernel)
        self.memory_kernel = nn.Parameter(torch.Tensor(hidden_size, order), requires_grad=trainable_memory_kernel)
        self.AT = nn.Parameter(torch.Tensor(self._A), requires_grad=trainable_A)
        self.BT = nn.Parameter(torch.Tensor(self._B), requires_grad=trainable_B)

        # Initialize parameters
        input_encoders_initializer(self.input_encoders)
        hidden_encoders_initializer(self.hidden_encoders)
        memory_encoders_initializer(self.memory_encoders)
        input_kernel_initializer(self.input_kernel)
        hidden_kernel_initializer(self.hidden_kernel)
        memory_kernel_initializer(self.memory_kernel)

    def forward(self, input, hx):

        h, m = hx

        u = (F.linear(input, self.input_encoders) +
             F.linear(h, self.hidden_encoders) +
             F.linear(m, self.memory_encoders))

        m = m + F.linear(m, self.AT) + F.linear(u, self.BT)

        h = self.hidden_activation(
            F.linear(input, self.input_kernel) +
            F.linear(h, self.hidden_kernel) +
            F.linear(m, self.memory_kernel))

        return h, [h, m]


class LegendreMemoryUnit(nn.Module):
    """
    Implementation of LMU using LegendreMemoryUnitCell so it can be used as LSTM or GRU in PyTorch Implementation (no GPU acceleration)
    """
    def __init__(self, input_dim, output_size, hidden_size, order, theta):
        super(LegendreMemoryUnit, self).__init__()

        self.hidden_size = hidden_size
        self.order = order

        self.lmucell = LMUCell(input_dim, hidden_size, order, theta)
        self.fc = nn.Linear(hidden_size, output_size)  # Added a fully connected layer to the net

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, xt, states):
        outputs = []

        if states == None:
        	h0 = torch.zeros(xt.size(0),self.hidden_size).to(self.device)
        	m0 = torch.zeros(xt.size(0),self.order).to(self.device)
        	states = (h0,m0)
        for i in range(xt.size(1)):
            out, states = self.lmucell(xt[:,i,:], states)
            outputs += [out]
        LMU_outputs = torch.stack(outputs).permute(1,0,2)
        p = self.fc(LMU_outputs)
        return p, states




#----------------------------------------------------CoRNN



class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self,x,hy,hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy),1)))
                                   - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class coRNN(nn.Module): 
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp,n_hid,dt,gamma,epsilon)
        self.readout = nn.Linear(n_hid, n_out)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, states):
    	x = x.permute(1,0,2) # changing the input shape from (batch_size, seq_len, features) to (seq_len, batch_size, features), doing this to fit into the situation.
    	outputs = []
    	## initialize hidden states

    	if states == None:
    		hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(self.device)
    		hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(self.device)
    	else:
    		hy, hz = states
    	for t in range(x.size(0)):
    		hy, hz = self.cell(x[t],hy,hz)
    		outputs.append(hy)

    	outputs = torch.stack(outputs)
    	outputs = outputs.permute(1,0,2) # changing the shape back to the original shape
    	output = self.readout(outputs)
    	return output, [hy, hz]