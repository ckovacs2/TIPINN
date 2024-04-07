import torch
import torch.nn as nn
import numpy as np
import torch_topological.nn as ttnn
numeric = int|float

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cubic0 = ttnn.CubicalComplex(dim = 0).to(DEVICE)
cubic1 = ttnn.CubicalComplex(dim = 1).to(DEVICE)
metric = ttnn.WassersteinDistance().to(DEVICE)

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
            return loss
        if loss and not weight:
            raise ValueError("Given loss but no loss weight. Please input loss weight if want to use extra loss function.")
    
    def to_grad(self, outputs, inputs):
        return torch.autograd.grad(
            outputs, inputs, grad_outputs=torch.ones_like(outputs), 
            create_graph=True
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
                loss += self.topo_loss_weight * self.topo_loss(self)

            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            out = self.forward(X)
        return out
    

class TIPINN_Cooling(TIPINN):
    def __init__(
        self,
        input_dim: int = 1, 
        hidden_dim:int = 200,
        output_dim:int = 1, 
        epochs=1000,
        loss=nn.MSELoss(),
        activation=nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-3,
        phys_loss = False,
        topo_loss = False,
        phys_loss_weight = 1,
        topo_loss_weight = 1,
        Tenv:float = 25,
        T0:int = 100,
        R:float = 0.005*3,
        noise_intensity: int = 2,
        dim = 0
    ) -> None:
        
        # Initial conditions and ground truths
        self.Tenv = Tenv
        self.T0 = T0
        self.R = R
        self.dim = dim
        self.times = np.linspace(0, 1000, 1000).reshape(-1, 1)
        self.temps = self.cooling_solution(time=self.times, Tenv=Tenv, T0=T0, R=R)
        self.times = torch.from_numpy(self.times).type(torch.float).to(DEVICE)

        # Make training data
        self.t = np.linspace(0, 300, 10).reshape(-1, 1)
        self.T = self.cooling_solution(time=self.t, Tenv=Tenv, T0=T0, R=R) +  noise_intensity * torch.normal(0, 1, size = [10, 1]).to(DEVICE)
        self.t = torch.from_numpy(self.t).type(torch.float).to(DEVICE)

        if phys_loss:
            phys_loss = self.phys_loss
        if topo_loss:
            topo_loss = self.topological_loss

        super().__init__(
            input_dim, 
            hidden_dim, 
            output_dim, 
            epochs, 
            loss, 
            activation,
            optimizer,
            lr,
            phys_loss,
            phys_loss_weight,
            topo_loss,
            topo_loss_weight,
        )

    def cooling_solution(self, time, Tenv, T0, R):
        T = Tenv + (T0 - Tenv) * np.exp(-R * time)
        return torch.from_numpy(T).type(torch.float).to(DEVICE)
    
    def cooling_pde(self, temp):
        return self.R * (self.Tenv - temp)

    def topological_loss(self, model):
        cubic = cubic1 if self.dim == 1 else cubic0
        ts = torch.linspace(0, 1000, steps=len(self.times)).view(-1,1).requires_grad_(True).to(DEVICE)
        temps = model(ts)

        # compute prediction diagram
        pde_dgm = cubic(self.cooling_pde(temps))

        # compute gradient diagram
        dT = self.to_grad(temps, ts)[0]
        grad_dgm = cubic(dT)

        # compute distances
        distances = [metric(pred, true) for pred, true in zip(grad_dgm, pde_dgm)]
        topo_loss = torch.stack(distances)
        distance = topo_loss.mean()
        return distance

    def phys_loss(self, model):
        ts = torch.linspace(0, 1000, steps=len(self.times)).view(-1,1).requires_grad_(True).to(DEVICE)
        temps = model(ts)
        dT = self.to_grad(temps, ts)[0]
        loss = torch.mean(torch.pow(self.cooling_pde(temps) - dT, 2))
        return loss