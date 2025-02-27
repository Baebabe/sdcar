import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_v4(nn.Module):
    def __init__(self, n_feature=5, n_hidden=64, n_output=2):
        super(Net_v4, self).__init__()
        self.hidden1 = nn.Linear(n_feature+1, n_hidden, bias=False)   
        self.hidden2 = nn.Linear(n_hidden, n_hidden, bias=False)   
        self.predict = nn.Linear(n_hidden, n_output, bias=False)   

    def forward(self, input_, training=False):
        v_sqrt = torch.sqrt(input_[:,0:1])
        beta = input_[:,1:2]
        steering = input_[:,2:3]
        throttle_brake = input_[:,3:5]

        x1 = torch.cat(
            (
                v_sqrt, 
                torch.cos(beta), 
                torch.sin(beta),
                steering,
                throttle_brake
            ),
            dim = -1
        )

        x2 = torch.cat(
            (
                v_sqrt, 
                torch.cos(beta), 
                -torch.sin(beta),
                -steering,
                throttle_brake
            ),
            dim = -1
        )

        x1 = torch.tanh(self.hidden1(x1))
        x1 = torch.tanh(self.hidden2(x1))
        x1 = self.predict(x1)            
        v_sqrt_dot1 = x1[:,0].unsqueeze(1)
        beta_dot1 = x1[:,1].unsqueeze(1)

        x2 = torch.tanh(self.hidden1(x2))     
        x2 = torch.tanh(self.hidden2(x2))  
        x2 = self.predict(x2)            
        v_sqrt_dot2 = x2[:,0].unsqueeze(1)
        beta_dot2 = x2[:,1].unsqueeze(1)

        x = torch.cat(
            (
                v_sqrt_dot1*(2*v_sqrt+v_sqrt_dot1) + v_sqrt_dot2*(2*v_sqrt+v_sqrt_dot2), 
                beta_dot1 - beta_dot2
            ), 
            dim = -1
        )/2

        return x