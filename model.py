import torch
import torch.nn as nn
import torch_topological.nn as ttnn
numeric = int|float

cubic = ttnn.CubicalComplex(dim = 0)
wasser = ttnn.WassersteinDistance()

class TIPINN(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim:int,
        output_dim:int, 
        epochs=1000,
        loss=nn.MSELoss(),
        activation=nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-3,
        phys_loss = None, 
        phys_loss_weight = None,
        topo_loss = False,
        topo_loss_weight = None
    ):
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.activation = activation
        self.lr = lr
        self.optimizer = optimizer
        self.phys_loss = phys_loss
        self.phys_loss_weight = phys_loss_weight
        self.topo_loss = topo_loss
        self.topo_loss_weight = topo_loss_weight

        # ensure weight is given if loss is given
        self.validate_loss(self.phys_loss, self.phys_loss_weight)
        self.validate_loss(self.topo_loss, self.topo_loss_weight)
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def validate_loss(self, loss, weight):
        if not loss:
            return 
        
        if loss and not weight:
            raise ValueError("Given loss but no loss weight. Please input loss weight if want to use extra loss function.")
    
    def to_grad(self, outputs, inputs):
        return torch.autograd.grad(
            outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
        )
    
    def forward(self, X):
        X = self.layers(X)
        X = self.out(X)
        return X
    
    def fit(self, X, y):
        optimiser = self.optimizer(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(X)
            loss = self.loss(y, outputs)

            if self.phys_loss:
                loss += self.phys_loss_weight + self.phys_loss_weight * self.phys_loss(self)

            if self.topo_loss:
                loss += self.topo_loss_weight + self.topo_loss_weight * self.topo_loss(self)

            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(X)
        return out
    

class TIPINN_Cooling(TIPINN):
    def __init__(
        self,
        input_dim: int = 1, 
        hidden_dim:int = 200,
        output_dim:int = 10, 
        epochs=1000,
        loss=nn.MSELoss(),
        activation=nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-3,
        phys_loss_weight = 1,
        topo_loss_weight = 1
    ) -> None:
        self.Tenv = 25
        self.T0 = 100
        self.R = 0.005
        self.times = torch.linspace(0, 1000, 1000)
        self.temps = self.cooling_law(time=self.times, Tenv=self.Tenv, T0=self.T0, R=self.R)

        # Make training data
        self.t = torch.linspace(0, 300, 10)
        self.T = self.cooling_law(time=self.t, Tenv=self.Tenv, T0=self.T0, R=self.R) +  2 * torch.normal(0, 1, size = [10])
        self.cubic_temps = cubic(self.temps)

        super().__init__(
            input_dim, 
            hidden_dim, 
            output_dim, 
            epochs, 
            loss, 
            activation,
            optimizer,
            lr,
            self.phys_loss(TIPINN(input_dim, hidden_dim, output_dim)),
            phys_loss_weight,
            self.topological_loss(TIPINN(input_dim, hidden_dim, output_dim)),
            topo_loss_weight
        )

    def cooling_law(self, time, Tenv, T0, R):
        T = Tenv + (T0 - Tenv) * torch.exp(-R * time)
        return T

    def topological_loss(self, model):
        ts = torch.linspace(0, 1000, steps=len(self.times)).view(-1,1).requires_grad_(True)
        temps = model(ts)
        dgm = cubic(temps)
        distance = wasser(self.cubic_temps, dgm)
        return distance

    def phys_loss(self, model):
        ts = torch.linspace(0, 1000, steps=len(self.times)).view(-1,1).requires_grad_(True)
        temps = model(ts)
        dT = self.to_grad(temps, ts)[0]
        pde = self.R*(self.Tenv - temps) - dT
        return torch.mean(pde**2)