import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from pathlib import Path
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()

"""
Training a fully forward NN (aka ANN) for regression of a highly non-linear and oscillatory 1D function. 

"""


def ff(x: float) -> float:
    T = 0.2
    return 4 * (x - 0.5) ** 3 + 0.5 + 0.2 * np.sin(2 * np.pi * x / T)


def make_data(
    f: Callable[[float], float], n, return_torch: bool = False
) -> tuple[float, float, torch.utils.data.TensorDataset]:
    x = np.random.rand(n).reshape(-1, 1)
    y = f(x)
    if not return_torch:
        return x, y, None
    xt = torch.FloatTensor(x)
    yt = torch.FloatTensor(y)
    dataset = torch.utils.data.TensorDataset(xt, yt)
    return x, y, dataset


def plot_constructor(
    f: Callable[[float], float], x1=None, y1=None, title: str = ""
) -> None:
    x = np.linspace(0, 1, 1001)
    y = f(x)
    plt.plot(x, y, label="Actual function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    if x1 is not None and y1 is not None:
        plt.scatter(x1, y1, color="red", marker="x", label="Model predictions")
    plt.title(title)
    plt.show()


class Net(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        layers: list[int] = [],
        output_dim: int = 1,
        activation: str = "ReLU",
        dropout_p: float = 0,
        use_initializer_uniform: bool = False,
        use_initializer_normal: bool = False,
    ):
        super(Net, self).__init__()
        layers.insert(0, input_dim)
        layers.append(output_dim)
        self.f = nn.ModuleList()
        self.activation = activation
        for i, (in_features, out_features) in enumerate(zip(layers[:-1], layers[1:])):
            self.f.append(nn.Linear(in_features=in_features, out_features=out_features))
            if use_initializer_uniform:
                if activation == "ReLu":
                    torch.nn.init.kaiming_uniform_(self.f[-1].weight)
                elif activation == "Tanh":
                    torch.nn.init.xavier_uniform_(self.f[-1].weight)
            elif use_initializer_normal:
                if activation == "ReLu":
                    torch.nn.init.kaiming_normal_(self.f[-1].weight)
                elif activation == "Tanh":
                    torch.nn.init.xavier_normal_(self.f[-1].weight)
            print(
                f"Added layer: nn.Linear(in_features={in_features}, out_features={out_features})"
            )
            if i < len(layers) - 1:
                self.f.append(nn.Tanh() if activation == "Tanh" else nn.ReLU())
                if dropout_p > 0.01:
                    self.f.append(nn.Dropout(p=dropout_p))

    def forward(self, x):
        for f in self.f:
            x = f(x)
        return x


def train(
    model: nn.Module,
    lr: float,
    dataloader_train: torch.utils.data.DataLoader,
    data_val: torch.utils.data.TensorDataset,
    epochs: int,
    patience: int = 100,
    plot_losses: bool = True,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    losses_train = []
    losses_val = []
    best_model_dict = None
    best_val_loss = None
    counter = 0
    for epoch in range(epochs):
        train_losses_epoch = []
        model.train()
        for x, y in dataloader_train:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            train_losses_epoch.append(loss.item())
        losses_train.append(np.average(np.array(train_losses_epoch)))
        model.eval()
        with torch.no_grad():
            x = data_val.tensors[0]
            y = data_val.tensors[0]
            yhat = model(x)
            loss = criterion(yhat, y)
            losses_val.append(loss.item())
            if best_val_loss is None or losses_val[-1] < best_val_loss:
                best_val_loss = losses_val[-1]
                best_model_dict = model.state_dict()
                counter = 1
            elif counter < patience:
                counter += 1
            else:
                break
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch = {epoch + 1}, train_loss = {losses_train[-1]}, val_loss = {losses_val[-1]}"
            )
    model.load_state_dict(best_model_dict)
    if plot_losses:
        plt.plot(losses_train, label="Training losses")
        plt.plot(losses_val, label="Validation losses")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.title(
            print(f"model parameters = {sum(p.numel() for p in model.parameters())}")
        )
        plt.show()
    return losses_train, losses_val


def eval_model(model: nn.Module, f: Callable[[float], float], n):
    x_test, y_test, dataset_test = make_data(f=f, n=n, return_torch=True)
    yhat = model(dataset_test.tensors[0]).detach().numpy()
    r2 = r2_score(y_true=y_test, y_pred=yhat)
    print(f"R^2 = {r2}")
    plot_constructor(f=f, x1=x_test, y1=yhat, title=f"R2 = {r2}")
    print("---------------------------------------------------------")


def main():
    batch_size = 500
    lr = 0.0001
    epochs = 10000
    n_train = 20000
    n_validation = 500
    n_test = 10000
    patience = 1000
    layers = [200, 50, 50]
    activation = "Tanh"
    dropout_p = 0
    use_initializer_uniform = False
    use_initializer_normal = False

    np.random.seed(42)
    torch.manual_seed(42)
    # x0, y0, _ = make_data(f=ff,n=10000)
    # plot_constructor(f=ff, x1=x0, y1=y0, title="")
    model = Net(
        input_dim=1,
        layers=layers,
        output_dim=1,
        activation=activation,
        dropout_p=dropout_p,
        use_initializer_uniform=use_initializer_uniform,
        use_initializer_normal=use_initializer_normal,
    )
    _, _, dataset_train = make_data(f=ff, n=n_train, return_torch=True)
    _, _, data_val = make_data(f=ff, n=n_validation, return_torch=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size
    )
    model_file = THIS_DIR / "regressor_model.pth"
    if model_file.is_file():
        model.load_state_dict(torch.load(model_file))
    else:
        train(
            model=model,
            lr=lr,
            dataloader_train=dataloader_train,
            data_val=data_val,
            epochs=epochs,
            patience=patience,
            plot_losses=True,
        )
        torch.save(model.state_dict(), model_file)
    eval_model(model=model, f=ff, n=n_test)


if __name__ == "__main__":
    main()

# py -m coursera.IBM_AI_ENG_PyTorch2.torch2_ann_regressor

"""
batch_size = 500
lr=0.001
epochs=10000
n_train = 20000
n_validation = 1000
n_test = 100
patience = 1000
layers = [10]
activation = "Tanh"
dropout_p = 0
use_initializer_uniform = False
use_initializer_normal = False
Epoch = 1050, train_loss = 0.018710127240046857, val_loss = 0.019638704136013985
R^2 = 0.63445260839987
---------------------------------------------------------
layers = [10]
Epoch = 480, train_loss = 0.018936258181929587, val_loss = 0.020710023120045662
model parameters = 31
R^2 = 0.6330140165868162
---------------------------------------------------------
n_validation = 20000
Epoch = 460, train_loss = 0.019300970993936063, val_loss = 0.020089559257030487
model parameters = 31
R^2 = 0.5433562462639191
---------------------------------------------------------
n_validation = 4000
layers = [100]
Epoch = 140, train_loss = 0.02030400247313082, val_loss = 0.020368026569485664
model parameters = 301
R^2 = 0.46121359780998494
---------------------------------------------------------
n_validation = 10000
layers = [1000]
Epoch = 200, train_loss = 0.01910765953361988, val_loss = 0.016610708087682724
model parameters = 3001
R^2 = 0.5238270395228911
---------------------------------------------------------
layers = [10000]        
Epoch = 100, train_loss = 0.2929944172501564, val_loss = 0.33218055963516235
model parameters = 30001
R^2 = -6.069282361242694
---------------------------------------------------------
layers = [2000]      
Epoch = 100, train_loss = 0.019787058373913168, val_loss = 0.01866036467254162
model parameters = 6001
R^2 = 0.5519696599461102
---------------------------------------------------------
n_test=10000
Epoch = 100, train_loss = 0.019787058373913168, val_loss = 0.01866036467254162
model parameters = 6001
R^2 = 0.5690651781328551
---------------------------------------------------------
layers = [1000, 1]
Epoch = 120, train_loss = 0.01903066486120224, val_loss = 0.020331943407654762
model parameters = 3003
R^2 = 0.5616681921996858
 ---------------------------------------------------------
layers = [1000, 2]
Epoch = 210, train_loss = 0.018412201665341854, val_loss = 0.020667269825935364
model parameters = 4005
R^2 = 0.574553666383071
---------------------------------------------------------
layers = [1000, 5]
Epoch = 110, train_loss = 0.01261331622954458, val_loss = 0.028801027685403824
model parameters = 7011
R^2 = 0.7114145169284523
---------------------------------------------------------
layers = [1000, 10]
Epoch = 100, train_loss = 0.004012748983222991, val_loss = 0.03273152932524681
model parameters = 12021
R^2 = 0.91212991354819     
---------------------------------------------------------
layers = [1000, 20]
Epoch = 100, train_loss = 0.0014762284728931264, val_loss = 0.03697732090950012
model parameters = 22041
R^2 = 0.9669881751772816
---------------------------------------------------------
layers = [1000, 50]  
Epoch = 100, train_loss = 0.0003839616598270368, val_loss = 0.03687116503715515
model parameters = 52101
R^2 = 0.9934295435168916
---------------------------------------------------------
lr=0.0001
Epoch = 110, train_loss = 0.016118353116326034, val_loss = 0.023634113371372223
model parameters = 52101
R^2 = 0.6362599605043975
---------------------------------------------------------
n_validation = 2000
Epoch = 120, train_loss = 0.015493146306835116, val_loss = 0.02450428530573845
model parameters = 52101
R^2 = 0.6417304176693688
---------------------------------------------------------
patience = 1000
Epoch = 1020, train_loss = 2.9231692366238347e-06, val_loss = 0.03895729035139084
model parameters = 52101
R^2 = 0.9999286571658339  **** Most Accurate ****
---------------------------------------------------------
layers = [500, 50]
Epoch = 1020, train_loss = 8.521902009306359e-06, val_loss = 0.038915906101465225
model parameters = 26101
R^2 = 0.9997955940791755   
Issue: validation does not improve and the program have to train within the patience epochs from start
Basically: Epochs to stop ~= patience!
---------------------------------------------------------
layers = [200, 50]
Epoch = 1060, train_loss = 0.0005210561452258844, val_loss = 0.03757172450423241
model parameters = 10501
R^2 = 0.9884828292769094
---------------------------------------------------------
n_validation = 500
layers = [500, 50]
Epoch = 1010, train_loss = 8.759047705098056e-06, val_loss = 0.04103123024106026
model parameters = 26101
R^2 = 0.9997910603205483 
---------------------------------------------------------
layers = [500, 50, 50]
Epoch = 1030, train_loss = 1.0780869683912896e-05, val_loss = 0.04071858152747154
model parameters = 28651
R^2 = 0.9996596190479294
---------------------------------------------------------
layers = [200, 50, 50]   ***** Optimum *****
Epoch = 1080, train_loss = 1.1463501357411587e-05, val_loss = 0.041147876530885696
model parameters = 13051
R^2 = 0.9997386399321934
---------------------------------------------------------
layers = [100, 50, 50]
Epoch = 1130, train_loss = 8.948570393840782e-05, val_loss = 0.04109087586402893
model parameters = 7851
R^2 = 0.9977223099399908
---------------------------------------------------------
layers = [50, 100, 50]
Epoch = 1190, train_loss = 5.1538921525207115e-05, val_loss = 0.040399231016635895
model parameters = 10301
R^2 = 0.9987258246142858
---------------------------------------------------------
layers = [50, 50, 100]
Epoch = 1220, train_loss = 0.008795229368843138, val_loss = 0.02784891240298748
model parameters = 7851
R^2 = 0.7964856478965217
---------------------------------------------------------
dropout_p = 0.3
Epoch = 1000, train_loss = 0.10235598348081112, val_loss = 0.05719687417149544
model parameters = 13051
R^2 = -0.10040256087013844
---------------------------------------------------------
layers = [200, 50, 50]   
use_initializer_uniform = True
Epoch = 1090, train_loss = 4.076694181094353e-05, val_loss = 0.0410030297935009
model parameters = 13051
R^2 = 0.9990678156558138
---------------------------------------------------------
layers = [200, 50, 50]   
use_initializer_uniform = False
use_initializer_normal = True
Epoch = 1100, train_loss = 9.869694781627914e-06, val_loss = 0.04104624316096306
model parameters = 13051
R^2 = 0.9998017348040606
=============================  ReLU ===============================
batch_size = 500
lr=0.001
epochs=10000
n_train = 20000
n_validation = 1000
n_test=100
patience = 1000
layers = []
activation="ReLU"
dropout_p = 0
use_initializer_uniform = False
use_initializer_normal = False

Epoch = 1140, train_loss = 0.02035466069355607, val_loss = 0.01902737468481064
model parameters = 2
R^2 = 0.4640691437676122
---------------------------------------------------------
layers = [1]
Epoch = 1690, train_loss = 0.02035601339302957, val_loss = 0.018948743119835854
model parameters = 4
R^2 = 0.4612126682793617
---------------------------------------------------------
layers = [5]

Epoch = 1010, train_loss = 0.019497850071638824, val_loss = 0.019419288262724876
model parameters = 16
R^2 = 0.49288153065718854
---------------------------------------------------------
layers = [20]
Epoch = 1000, train_loss = 0.29310308694839476, val_loss = 0.34691131114959717
model parameters = 61
R^2 = -6.445936479928191
---------------------------------------------------------
"""
