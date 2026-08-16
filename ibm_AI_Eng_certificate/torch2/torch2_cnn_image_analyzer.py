import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import copy

""" 
A CNN machine that can classify a line as vertical, horizontal, diagonal and anti-diagonal direction.
A separate program in this file can be used to create the training images and also put noise with a controlled percentage into the images.
These function were used to see at what percentage of noisy pixels the trained CNN fails to produce the correct classification.  
"""

THIS_PATH = Path(__file__).parent.resolve()


class CNN_Model(nn.Module):
    def __init__(self, image_size: tuple[int, int] = [20, 20]) -> None:
        super().__init__()
        out_channels_1 = 4
        out_channels_2 = 8
        cnn1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels_1,
            kernel_size=5,
            stride=1,
            padding=0,
        )
        cnn1_bn = nn.BatchNorm2d(out_channels_1)
        mp1 = nn.MaxPool2d(kernel_size=2)
        cnn2 = nn.Conv2d(
            in_channels=out_channels_1,
            out_channels=out_channels_2,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        cnn2_bn = nn.BatchNorm2d(out_channels_2)
        mp2 = nn.MaxPool2d(kernel_size=2)
        f0 = image_size[0]
        f1 = image_size[1]
        for layer in [cnn1, mp1, cnn2, mp2]:
            print(
                f"layer.kernel_size = {layer.kernel_size}, type = {type(layer.kernel_size) == tuple}"
            )
            k = (
                layer.kernel_size
                if type(layer.kernel_size) == int
                else layer.kernel_size[0]
            )
            s = layer.stride if type(layer.stride) == int else layer.stride[0]
            f0 = int((f0 - k) / s + 1)
            f1 = int((f1 - k) / s + 1)
        self.image_processing = nn.ModuleList()
        self.image_processing.append(cnn1)
        self.image_processing.append(cnn1_bn)
        self.image_processing.append(nn.ReLU())
        self.image_processing.append(mp1)
        self.image_processing.append(cnn2)
        self.image_processing.append(cnn2_bn)
        self.image_processing.append(nn.ReLU())
        self.image_processing.append(mp2)
        self.flat_size = f0 * f1 * out_channels_2
        self.ln = nn.Linear(in_features=self.flat_size, out_features=4)

        print(f"f0 = {f0}, f1 = {f1}, flat_size = {self.flat_size}")

    def forward(self, x):
        for f in self.image_processing:
            x = f(x)
        x = self.ln(x.view(-1, self.flat_size))
        return x

    def analysis(self, x):
        cnn1 = self.image_processing[0]
        mp1 = self.image_processing[3]
        cnn2 = self.image_processing[4]
        mp2 = self.image_processing[7]
        x_cnn1 = cnn1(x)
        x_mp1 = mp1(torch.relu(x_cnn1))
        x_cnn2 = cnn2(x_mp1)
        x_mp2 = mp2(torch.relu(x_cnn2))
        return x_cnn1, x_mp1, x_cnn2, x_mp2


def plot_mat(mat: np.ndarray):
    n = min(mat.shape[0], mat.shape[1])
    fig_mat = np.copy(mat[:n, :n])
    a = np.min(fig_mat)
    b = np.max(fig_mat)
    fig_mat = 255 * (1 - (fig_mat - a) / (b - a))
    img = Image.fromarray(fig_mat)
    plt.imshow(img)
    plt.show()


def gen_images(
    n_images: int, mode="v", noise_fraction=0.05, save_image: bool = True
) -> None:
    """
    Modes: vertical lines
    """
    img_numbers = []
    image_size = 20
    n_noise_points = int(image_size * image_size * noise_fraction)
    print(f"n_noise_points = {n_noise_points}")
    if save_image:
        df = pd.read_csv(THIS_PATH / "images_20x20/data/labels.csv")
    for _ in range(n_images):
        j = np.random.randint(17)
        mat = np.zeros([image_size, image_size], dtype=int)
        if mode == "v":
            mat[:, j : j + 4] = 1
        elif mode == "h":
            mat[j : j + 4, :] = 1
        elif mode == "d" or mode == "d2":
            ind = np.diag_indices(20)
            offset = np.random.randint(31) - 15
            for s in range(offset - 2, offset + 3):
                if s > 0:
                    mat[ind[0][: -abs(s)], ind[1][: -abs(s)] + abs(s)] = 1
                elif s < 0:
                    mat[ind[0][: -abs(s)] + abs(s), ind[1][: -abs(s)]] = 1
                else:
                    mat[ind] = 1
            if mode == "d2":
                mat = np.rot90(mat)
        ind = np.random.randint(image_size, size=[n_noise_points, 2])
        mat[ind[:, 0], ind[:, 1]] = 1
        n = min(mat.shape[0], mat.shape[1])
        fig_mat = np.copy(mat[:n, :n])
        a = np.min(fig_mat)
        b = np.max(fig_mat)
        fig_mat = 255 * (1 - (fig_mat - a) / (b - a))
        img = Image.fromarray(fig_mat.astype(np.uint8))
        img_number = np.random.randint(1000000)
        img_numbers.append(img_number)
        if save_image:
            img_name = str(img_number) + ".png"
            img_file = THIS_PATH / (f"images_20x20/pngs/{img_name}")
            img.save(img_file)
            df.loc[len(df)] = [img_number, mode]
        else:
            return img
    if save_image:
        df.to_csv(THIS_PATH / "images_20x20/data/labels.csv", index=False)


def sample_images() -> None:
    mat = np.zeros([20, 20])
    mat[:, 0:4] = 1
    # plot_mat(mat)

    mat = np.zeros([20, 20])
    mat[:, 6:10] = 1
    # plot_mat(mat)

    mat = np.zeros([20, 20])
    mat[:, 16:20] = 1
    # plot_mat(mat)

    mat = np.zeros([20, 20])
    mat[6:10, :] = 1
    # plot_mat(mat)

    mat = np.zeros([20, 20])
    ind = np.diag_indices(20)
    mat[ind] = 1
    mat[ind[0][:-1], ind[0][:-1] + 1] = 1
    mat[ind[0][:-2], ind[0][:-2] + 2] = 1
    mat[ind[0][:-1] + 1, ind[0][:-1]] = 1
    mat[ind[0][:-2] + 2, ind[0][:-2]] = 1
    # plot_mat(mat)
    # plot_mat(np.rot90(mat))

    mat = np.zeros([20, 20])
    ind = np.diag_indices(20)
    offset = -5
    for s in range(offset - 2, offset + 3):
        if s > 0:
            mat[ind[0][: -abs(s)], ind[1][: -abs(s)] + abs(s)] = 1
        elif s < 0:
            mat[ind[0][: -abs(s)] + abs(s), ind[1][: -abs(s)]] = 1
        else:
            mat[ind] = 1
    # plot_mat(mat)
    plot_mat(np.rot90(mat))
    # plot_mat(mat.T)
    # plot_mat(np.rot90(mat.T))


def generate_images() -> None:
    df = pd.DataFrame(columns=["image_number", "label"])
    df.to_csv(THIS_PATH / "images_20x20/data/labels.csv", index=False)
    np.random.seed(42)
    gen_images(n_images=2500, mode="v", noise_fraction=0.05)
    gen_images(n_images=2500, mode="h", noise_fraction=0.05)
    gen_images(n_images=2500, mode="d", noise_fraction=0.05)
    gen_images(n_images=2500, mode="d2", noise_fraction=0.05)


def load_data(data_len: int = -1, verbose: bool = False):
    df = pd.read_csv(THIS_PATH / "images_20x20/data/labels.csv")
    df = df.drop_duplicates(subset=["image_number"], keep="last")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_encoded = pd.get_dummies(df, columns=["label"])
    y = df_encoded[["label_d", "label_d2", "label_h", "label_v"]]
    df[["label_d", "label_d2", "label_h", "label_v"]] = y
    df.to_csv(THIS_PATH / "images_20x20/data/labels_cleaned.csv")
    y = y if data_len == -1 else y[:data_len]
    labels = df["label"] if data_len == -1 else df["label"].values[:data_len]
    n_data = len(df) if data_len == -1 else data_len
    d0 = 20
    d1 = 20
    x = np.zeros([n_data, 1, d0, d1])
    t0 = time.time()
    for i in range(n_data):
        img_n = df.iloc[i, 0]
        image_file = THIS_PATH / "images_20x20/pngs" / (f"{img_n}.png")
        img = Image.open(image_file)
        x_img = np.array(img)
        x[i, 0, :, :] = x_img[:, :]
        if (i + 1) % 100 == 0:
            print(f"Image {i + 1} out of {n_data} is done.")
    t1 = time.time()
    if verbose:
        print(
            f"x size = {x.nbytes / (1024**2)} MB, time to create {n_data} data: {round(t1 - t0)} sec"
        )
        print(f"x.shape = {x.shape}")
        print(df.sample(10))
        print(f"df_encoded = {df_encoded.sample(5)}")
    """
    x size = 23329.6875 MB, time to create 9954 data: 39 sec
    """
    return x, y, labels


def dev_cnn(retrain: bool = False):
    data_len = 5000
    batch_size = data_len // 20
    epochs = 100000
    lr = 0.001
    patience = 10
    model_file = THIS_PATH / "images_20x20/data/cnn_model.pth"
    best_val_loss = None
    x0, y0, labels = load_data(data_len=data_len)
    x_train, x2, y_train, y2, labels_train, labels2 = train_test_split(
        x0, y0, labels, random_state=0, test_size=0.2, stratify=labels
    )
    x_val, x_test, y_val, y_test, labels_val, labels_test = train_test_split(
        x2, y2, labels2, random_state=0, test_size=0.5, stratify=labels2
    )
    torch.manual_seed(42)
    dataset_train = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train.values)
    )
    dataset_val = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val.values)
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test.values)
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size
    )

    model = CNN_Model()
    criterion = nn.BCEWithLogitsLoss()
    if not retrain and model_file.is_file():
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        losses_train = []
        losses_val = []
        accuracies_train = []
        accuracies_val = []
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            for x, y in dataloader_train:
                optimizer.zero_grad()
                yhat = model(x)
                loss = criterion(yhat, y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                x_train = dataset_train.tensors[0]
                yhat_train = model(x_train)
                yhat_train_softmax = torch.softmax(yhat_train, dim=1)
                yhat_train_max_ind = torch.argmax(yhat_train_softmax, dim=1)
                yhat_train_hot = torch.nn.functional.one_hot(
                    yhat_train_max_ind, num_classes=yhat_train_softmax.shape[-1]
                )
                y_train = dataset_train.tensors[1]
                losses_train.append(criterion(yhat_train, y_train).item())
                accuracy_train = (
                    torch.sum(yhat_train_hot == y_train).item()
                    / len(yhat_train)
                    * 100
                    / 4
                )
                accuracies_train.append(accuracy_train)

                x_val = dataset_val.tensors[0]
                yhat_val = model(x_val)
                yhat_val_softmax = torch.softmax(yhat_val, dim=1)
                yhat_val_max_ind = torch.argmax(yhat_val_softmax, dim=1)
                yhat_val_hot = torch.nn.functional.one_hot(
                    yhat_val_max_ind, num_classes=yhat_val_softmax.shape[-1]
                )
                y_val = dataset_val.tensors[1]
                losses_val.append(criterion(yhat_val, y_val).item())
                accuracy_val = (
                    torch.sum(yhat_val_hot == y_val).item() / len(yhat_val) * 100 / 4
                )
                accuracies_val.append(accuracy_val)
                # if epoch == 9:
                #     print(f"yhat_hot = \n{yhat_val_hot}")
                #     print(f"y_val = \n{y_val}")
                #     return
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch: {epoch + 1}: losses_train = {losses_train[-1]}, losses_val = {losses_val[-1]}"
                        + f", acc_train = {accuracies_train[-1]}%, acc_val = {accuracies_val[-1]}%"
                    )
                if best_val_loss is None or losses_val[-1] < best_val_loss:
                    best_val_loss = losses_val[-1]
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                elif patience_counter > patience:
                    break
                patience_counter += 1
        torch.save(best_model_state, model_file)
        plt.subplot(2, 1, 1)
        plt.plot(losses_train, label="Training")
        plt.plot(losses_val, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(accuracies_train, label="Training")
        plt.plot(accuracies_val, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accurcay%")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Epochs needed = {epoch}")
        plt.tight_layout()
        plt.savefig(THIS_PATH / "images_20x20/data/accuracies.png")
        plt.show()
    yhat_test = model(dataset_test.tensors[0])
    loss_test = criterion(yhat_test, dataset_test.tensors[1]).item()
    print(f"loss_test = {loss_test}")

    probs = torch.softmax(yhat_test, dim=1)
    zhat_test = torch.argmax(probs, dim=1).detach().numpy()
    print(f"zhat_test = {zhat_test}")
    z_test = (
        y_test.values.astype(int)[:, 1]
        + y_test.values.astype(int)[:, 2] * 2
        + y_test.values.astype(int)[:, 3] * 3
    )
    print(f"z_test = {z_test}")
    cm = confusion_matrix(zhat_test, z_test)
    print(f"cm = {cm}")


def test_image(n_samples=6, seed=42):
    np.random.seed(seed)
    model = CNN_Model()
    model.eval()
    model_file = THIS_PATH / "images_20x20/data/cnn_model.pth"
    modes = ["d", "d2", "h", "v"]
    if model_file.is_file():
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    else:
        return
    noise_fractions = [0.05, 0.05, 0.05, 0.05, 0.075, 0.1, 0.2, 0.5, 0.7, 0.8]
    for n in range(n_samples):
        mode = modes[n] if n < len(modes) else np.random.choice(modes)
        noise_fraction = (
            noise_fractions[n] if n < len(noise_fractions) else noise_fractions[-1]
        )
        img = gen_images(
            n_images=1, mode=mode, noise_fraction=noise_fraction, save_image=False
        )
        img = np.array(img)
        print(f"Image size = {img.shape}")
        title = f"sample #{n + 1}"
        title += f", label = '{mode}'"

        imgt = torch.FloatTensor(img[:, :])

        d0 = int(imgt.shape[0])
        d1 = int(imgt.shape[1])
        imgt = imgt.view(1, 1, d0, d1)
        prediction = model(imgt)
        prediction_ind = torch.argmax(torch.softmax(prediction, dim=1)).item()
        prediction_txt = modes[prediction_ind]
        title += f"\nnoise fraction = {noise_fraction}, prediction = '{prediction_txt}'"
        plt.imshow(img[:, :])
        plt.title(title)
        plt.tight_layout()
        plt.show()

        x = model.analysis(imgt)
        print(f"sizes: ")
        print(f" ****** imgt shape = {imgt.shape}")
        print(f" x0 = cnn1(imgt), x0.shape = {x[0].shape}")
        print(f" x1 = mp1(x0), x1.shape = {x[1].shape}")
        print(f" x2 = cnn2(x1), x2.shape = {x[2].shape}")
        print(f" x3 = mp2(x2), x3.shape = {x[3].shape}")
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(x[0][0, i, :, :].detach().numpy())
        plt.show()
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(x[1][0, i, :, :].detach().numpy())
        plt.show()
        for i in range(8):
            plt.subplot(4, 2, i + 1)
            plt.imshow(x[2][0, i, :, :].detach().numpy())
        plt.show()
        for i in range(8):
            plt.subplot(4, 2, i + 1)
            plt.imshow(x[3][0, i, :, :].detach().numpy())
        plt.show()


def plot_filters():
    model = CNN_Model()
    model.eval()
    model_file = THIS_PATH / "images_20x20/data/cnn_model.pth"
    if model_file.is_file():
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    else:
        return
    cnn1 = model.image_processing[0].weight
    print(f"cnn1.shape = {cnn1.shape}")
    for i in range(4):
        fig_mat = cnn1[i, 0, :, :].detach().numpy()
        plt.subplot(2, 2, i + 1)
        plt.imshow(fig_mat)
    plt.suptitle("CNN1")
    plt.show()

    cnn2 = model.image_processing[4].weight
    print(f"cnn2.shape = {cnn2.shape}")
    for i in range(8):
        for j in range(4):
            fig_mat = cnn2[i, j, :, :].detach().numpy()
            plt.subplot(8, 4, i * 4 + j + 1)
            plt.imshow(fig_mat)
    plt.suptitle("CNN2")
    plt.show()

    modes = ["d", "d2", "h", "v"]
    lin = model.ln.weight.detach().numpy()
    print(f"lin.shape = {lin.shape}")
    lin = lin.reshape(4, 8, 3, 3)
    j = 0
    for m in range(4):
        for i in range(8):
            j += 1
            plt.subplot(4, 8, j)
            plt.imshow(lin[m, i, :, :])
            plt.title(modes[m])
    plt.tight_layout()
    plt.show()


def main():
    generate_images()
    dev_cnn()
    test_image(n_samples=10)
    plot_filters()


if __name__ == "__main__":
    main()
    print("====== DONE ======")

# py -m coursera.IBM_AI_ENG_PyTorch2.torch2_cnn_image_analyzer
"""
cm = [
 [122   0   0   0]
 [  0 122   0   0]
 [  0   0 127   0]
 [  0   0   0 129]
 ]
 """
