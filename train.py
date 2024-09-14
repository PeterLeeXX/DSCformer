import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from model import dsc
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker

n_epoch = 5
Num_Workers = 1
BATCH_SIZE = 8
random_seed = 0
original_learn_rate = 1e-3
Patch_Size = 60
Depth = 1
Devide = 1
mlp_ratio = 1.
IF_Train_Model = True

def _init_vit_weights(m):
    """
    weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def seed_torch(seed=random_seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.spec_data = data
        self.spec_labels = label

    def __len__(self):
        return len(self.spec_labels)

    def __getitem__(self, idx):
        spec_piece = self.spec_data[idx, :]
        label_piece = self.spec_labels.iloc[idx]
        _spec = torch.tensor(spec_piece)
        _lable = torch.tensor(label_piece)

        spec = _spec.unsqueeze(-2)
        label = _lable.unsqueeze(-1)

        return spec, label


def savitzkygolay_filter(data):
    # S-G smoothing filter function
    window_width = 49
    polyorder = 2
    return savgol_filter(data, window_width, polyorder)


def train(model):

    """Record the number of training sessions"""
    total_train_step = 0
    """Record the number of tests"""
    total_test_step = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=original_learn_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-8)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
    #                                                           verbose=True, min_lr=1e-8)
    loss_func = RMSELoss()
    writer = SummaryWriter(log_dir='/root/tf-logs')
    writer.add_graph(model=model, input_to_model=torch.randn(1, 1, 4200).to(device))
    progress_bar = tqdm(range(n_epoch), desc="epoch", leave=True, colour='green', ncols=140)

    for _ in progress_bar:

        model.train()

        for batch_idx, batch in enumerate(data_train_loader):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            writer.add_scalar("train_loss", loss.item(), total_train_step)

        lr_scheduler.step(loss.item())

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_val_loader):
                test_x, test_y = tuple(t.to(device) for t in batch)
                pred_test = model(test_x)
                loss = loss_func(pred_test, test_y)
                test_loss = loss.item()

                test_label = test_y.cpu()
                pred_label = pred_test.cpu()

                r2 = r2_score(test_label, pred_label)

                rpd = (test_label.std() / loss.item()).item()

        writer.add_scalar('RMSE', test_loss, total_test_step)
        writer.add_scalar('R2', r2, total_test_step)
        writer.add_scalar('RPD', rpd, total_test_step)
        progress_bar.set_postfix({
            'test_loss': '{:.3f}'.format(test_loss),
            'R2': '{:.3f}'.format(r2),
            'RPD': '{:.3f}'.format(rpd),
            'lr=': '{:}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        })
        total_test_step = total_test_step + 1


def predict(model):

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_test_loader):
            test_x, test_y = tuple(t.to(device) for t in batch)
            output = model(test_x)
            test_label = test_y.cpu()
            output_label = output.cpu()

            mse = mean_squared_error(test_label, output_label)
            rmse = np.sqrt(mse)
            rpd = (test_label.std() / rmse).item()
            r2 = r2_score(test_label, output_label)

    print("RMSE on the overall test set: {}".format(rmse))
    print("R2   on the overall test set: {}".format(r2))
    print("RPD  on the overall test set: {}".format(rpd))


if __name__ == "__main__":
    seed_torch()  # Fixed random number seeds
    file_path = "./test data/LUCAS.SOIL_test.csv"    # Read Data
    Original_Data = pd.read_csv(file_path, low_memory=False, dtype='float32')
    LUCAS = Original_Data.iloc[:, :]

    label = LUCAS.loc[:, 'N']
    data = LUCAS.iloc[:, :4200]
    data_process = savitzkygolay_filter(data)
    data_train, data_val, label_train, label_val = train_test_split(data_process, label, test_size=0.1,
                                                                    random_state=random_seed)
    data_train, data_test, label_train, label_test = train_test_split(data_train, label_train, test_size=0.11111,
                                                                      random_state=random_seed)

    data_train_set = SpectralDataset(data_train, label_train)
    data_train_loader = torch.utils.data.DataLoader(data_train_set, batch_size=BATCH_SIZE, shuffle=True,
                                                    num_workers=Num_Workers, drop_last=True)
    data_val_set = SpectralDataset(data_val, label_val)
    data_val_loader = torch.utils.data.DataLoader(data_val_set, batch_size=len(label_val), shuffle=True,
                                                  num_workers=Num_Workers, drop_last=False)
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Train Set：", label_train.describe(), label_train.skew())
    print("Test Set：", label_test.describe(), label_test.skew())
    print("Val Set：", label_val.describe(), label_val.skew())

    model = dsc(patch_size=60, depth=1, p_divide=5, mlp_ratio=1.0, token_mixer='dsc')
    _init_vit_weights(model)
    model.to(device)
    if IF_Train_Model:
        train(model)
        torch.save(model.state_dict(), './params/DSCformer.pth')

    else:
        model.load_state_dict(torch.load('./params/DSCformer.pth'))
        model.to(device)

    DATA = data_test
    LABEL = label_test
    data_set = SpectralDataset(DATA, LABEL)
    data_test_loader = torch.utils.data.DataLoader(data_set, batch_size=len(LABEL), shuffle=False, drop_last=False)
    predict(model)
