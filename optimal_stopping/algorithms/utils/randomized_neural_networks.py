import torch



def init_weights(m, mean=0., std=1.):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, mean, std)
        torch.nn.init.normal_(m.bias, mean, std)


def init_weights_gen(mean=0., std=1., mean_b=0., std_b=1., dist=0):
    def init_weights(m, mean=mean, std=std, mean_b=mean_b, std_b=std_b,
                     dist=dist):
        if type(m) == torch.nn.Linear:
            if dist == 0:
                torch.nn.init.normal_(m.weight, mean, std)
                torch.nn.init.normal_(m.bias, mean_b, std_b)
            elif dist == 1:
                torch.nn.init.uniform_(m.weight, mean, std)
                torch.nn.init.uniform_(m.bias, mean_b, std_b)
            elif dist == 2:
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.bias)
            elif dist == 3:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.xavier_normal_(m.bias)
            else:
                raise ValueError
    return init_weights



class Reservoir(torch.nn.Module):
    def __init__(self, hidden_size, state_size, factors=(1.,),
                 activation=torch.nn.LeakyReLU(0.5)):
        super().__init__()
        self.factors = factors
        self.hidden_size = hidden_size
        layers = [
            torch.nn.Linear(state_size, hidden_size, bias=True),
            activation
        ]
        self.NN = torch.nn.Sequential(*layers)
        if len(factors) == 8:
            _init_weights = init_weights_gen(*factors[3:])
        else:
            _init_weights = init_weights
        self.apply(_init_weights)

    def forward(self, input):
        return self.NN(input*self.factors[0])



class randomRNN(torch.nn.Module):
    def __init__(self, hidden_size, state_size, factors=(1.,1.,1.),
                 extend=False):
        super().__init__()
        self.factors = factors
        self.extend = extend
        if self.extend:
            self.hidden_size = int(hidden_size/2)
        else:
            self.hidden_size = hidden_size
        layers = [
            torch.nn.Linear(state_size + self.hidden_size, self.hidden_size,
                            bias=True),
            torch.nn.Tanh()
        ]
        self.NN = torch.nn.Sequential(*layers)
        if self.extend:
            layers = [
                torch.nn.Linear(state_size, self.hidden_size, bias=True),
                torch.nn.Tanh()
            ]
            self.NN2 = torch.nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, input):
        h = torch.zeros(input.shape[1], self.hidden_size)
        if self.extend:
            hs_size = list(input.shape[:-1]) + [self.hidden_size*2]
            hs = torch.zeros(hs_size)
            for i in range(input.shape[0]):
                x = torch.cat([torch.tanh(input[i]*self.factors[0]),
                               h*self.factors[1]], dim=-1)
                h = self.NN(x)
                hs[i] = torch.cat([h, self.NN2(input[i]*self.factors[2])],
                                  dim=-1)

        else:
            hs_size = list(input.shape[:-1]) + [self.hidden_size]
            hs = torch.zeros(hs_size)
            for i in range(input.shape[0]):
                x = torch.cat([input[i]*self.factors[0], h*self.factors[1]],
                              dim=-1)
                h = self.NN(x)
                hs[i] = h
        return hs

