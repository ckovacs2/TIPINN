import torch
import torch.nn as nn
import numpy as np
import torch_topological.nn as ttnn
numeric = int|float

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cubic = ttnn.CubicalComplex().to(DEVICE)
cubic.training = False
metric = ttnn.WassersteinDistance().to(DEVICE)
metric.training = False

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
        topo_loss = None,
        topo_loss_weight = None,
        ols_loss = None,
        ols_loss_weight = None
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
        self.ols_loss = ols_loss
        self.ols_loss_weight = ols_loss_weight

        # ensure weight is given if loss is given
        self.validate_loss(self.phys_loss, self.phys_loss_weight)
        self.validate_loss(self.topo_loss, self.topo_loss_weight)
        self.validate_loss(self.ols_loss, self.ols_loss_weight)
        
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

            if self.ols_loss:
                loss += self.ols_loss_weight * self.ols_loss(self)

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
        ols_loss = False,
        phys_loss_weight = 1,
        topo_loss_weight = 1,
        ols_loss_weight = 1,
        Tenv:float = 25,
        T0:int = 100,
        R:float = 0.005*3,
        noise_intensity: int = 2,
    ) -> None:
        
        # Initial conditions and ground truths
        self.Tenv = Tenv
        self.T0 = T0
        self.R = R

        # Ground Truth
        self.true_times = torch.linspace(0, 1000, 1000).view(-1, 1).to(DEVICE)
        self.true_temps = self.cooling_solution(time=self.true_times, Tenv=Tenv, T0=T0, R=R)

        # Make training data
        self.approx_times = torch.linspace(0, 300, 10).view(-1, 1).to(DEVICE)
        self.approx_temps = self.cooling_solution(time=self.approx_times, Tenv=Tenv, T0=T0, R=R) +  noise_intensity * torch.normal(0, 1, size = [10, 1]).to(DEVICE)

        if phys_loss:
            phys_loss = self.phys_loss
        if topo_loss:
            topo_loss = self.topological_loss
        if ols_loss:
            ols_loss = self.ols_loss

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
            ols_loss,
            ols_loss_weight
        )

    def cooling_solution(self, time, Tenv, T0, R):
        T = Tenv + (T0 - Tenv) * torch.exp(-R * time)
        return T
    
    def cooling_pde(self, temp):
        return self.R * (self.Tenv - temp)

    def topological_loss(self, model):
        ts = torch.linspace(0, 1000, steps=len(self.true_times)).view(-1,1).requires_grad_(True).to(DEVICE)
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
        ts = torch.linspace(0, 1000, steps=len(self.true_times)).view(-1,1).requires_grad_(True).to(DEVICE)
        temps = model(ts)
        dT = self.to_grad(temps, ts)[0]
        loss = torch.mean(torch.pow(self.cooling_pde(temps) - dT, 2))
        return loss
    
    def ols_loss(self, model):
        return torch.sum(sum([p.pow(2.) for p in model.parameters()]))
    
class TIPINN_Oscillation(TIPINN):
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
        ols_loss = False,
        topo_loss_weight = 1,
        ols_loss_weight = 1,
        d:float = 2,
        w0:int = 20,
    ) -> None:
        
        # Initial conditions and ground truths
        self.d = d
        self.w0 = w0
        self.mu = 2*d
        self.k = w0**2
    
        # Ground Truth
        self.true_space = torch.linspace(0,1,500).view(-1,1)
        self.true_oscillate = self.oscillation_solution(d = self.d, w0=self.w0, x = self.x).to(DEVICE)
        # define boundary points, for the boundary loss
        self.t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)

        # define training points over the entire domain, for the physics loss
        self.t_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)

        # Make training data
        self.approx_space = self.true_space[0:200:20]
        self.approx_temps = self.true_oscillate[0:200:20]

        if phys_loss:
            phys_loss = self.phys_loss
        if topo_loss:
            topo_loss = self.topological_loss
        if ols_loss:
            ols_loss = self.ols_loss

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
            None,
            topo_loss,
            topo_loss_weight,
            ols_loss,
            ols_loss_weight
        )

    def oscillation_solution(self, d, w0, x):
        """Defines the analytical solution to the 1D underdamped harmonic oscillator problem.
        Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
        assert d < w0
        w = np.sqrt(w0**2-d**2)
        phi = np.arctan(-d/w)
        A = 1/(2*np.cos(phi))
        cos = torch.cos(phi+w*x)
        sin = torch.sin(phi+w*x)
        exp = torch.exp(-d*x)
        y  = exp*2*A*cos
        return y
    
    def oscillation_pde(self, d2udt2, dudt, u):
        return d2udt2 + self.mu*dudt + self.k*u

    def topological_loss(self, model):
        """Notice that for the topological loss, we are only comparing the recovery of the pde without considering boundaries.
        """
        ts = torch.linspace(0, self.true_space.max(), steps=len(self.true_space)).view(-1,1).requires_grad_(True).to(DEVICE)
        u = model(ts)
        dudt = self.to_grad(u, ts)[0]
        d2udt2 = self.to_grad(dudt, ts)[0]
        pde_dgm = cubic(self.oscillation_pde(d2udt2, dudt, u))

        # compute gradient diagram
        dudt_recovered = self.to_grad(u, ts)[0]
        d2udt2_recovered = self.to_grad(dudt_recovered, ts)[0]
        grad_dgm = cubic(d2udt2_recovered)

        # compute distances
        distances = [metric(pred, true) for pred, true in zip(grad_dgm, pde_dgm)]
        topo_loss = torch.stack(distances)
        distance = topo_loss.mean()
        return distance

    def phys_loss(self, model):
        # compute each term of the PINN loss function above
        # using the following hyperparameters:
        lambda1, lambda2 = 1e-1, 1e-4

        # compute boundary loss
        u = model(self.t_boundary)
        loss1 = (torch.squeeze(u) - 1)**2
        dudt = self.to_grad(u, self.t_boundary)[0]
        loss2 = (torch.squeeze(dudt) - 0)**2

        # compute physics loss
        u = model(self.t_physics)
        dudt = torch.autograd.grad(u, self.t_physics, torch.ones_like(u), create_graph=True)[0]
        d2udt2 = torch.autograd.grad(dudt, self.t_physics, torch.ones_like(dudt), create_graph=True)[0]
        loss3 = torch.mean((d2udt2 + self.mu*dudt + self.k*u)**2)

        # backpropagate joint loss, take optimiser step
        loss = loss1 + lambda1*loss2 + lambda2*loss3
        return loss
    
    def ols_loss(self, model):
        return torch.sum(sum([p.pow(2.) for p in model.parameters()]))
    
    def fit(self, X, y):
        optimiser = self.optimizer(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(X)
            if self.phys_loss:
                loss = self.phys_loss(self)
            else:
                loss = self.loss(y, outputs)
                if self.topo_loss:
                    loss += self.topo_loss_weight * self.topo_loss(self)

                if self.ols_loss:
                    loss += self.ols_loss_weight * self.ols_loss(self)

            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses