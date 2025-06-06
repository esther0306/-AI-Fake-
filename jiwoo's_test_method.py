import librosa
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os
from sklearn.metrics import roc_auc_score
from torchvision import transforms
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class Config:
    SR = 32000
    N_FEAT = 13
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 16
    N_EPOCHS = 5
    LR = 0.001
    SEED = 42
    
CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)

df = pd.read_csv('/home/work/.jun/contest/team_2/train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

def get_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        features.append(row['path'])
        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)
    if train_mode:
        return features, labels
    return features

train_feat, train_labels = get_feature(train, True)
val_feat, val_labels = get_feature(val, True)

class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, waveform):
        max_offset = waveform.size(1) - self.crop_size
        offset = random.randint(0, max_offset)
        return waveform[:, offset:offset + self.crop_size]

class CustomDataset(Dataset):
    def __init__(self, feat, label, sr, transform):
        self.feat = feat
        self.label = label
        self.sr = sr
        self.transform = transform
        self.silent_samples = []  # 무음 신호 샘플의 경로를 저장할 리스트

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, index):
        waveform, sr = torchaudio.load(self.feat[index])
        desired_length = 32000

        if waveform.size(1) > desired_length:
            max_start = waveform.size(1) - desired_length
            start = torch.randint(0, max_start, (1,)).item()
            waveform = waveform[:, start:start + desired_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, 32000 - waveform.shape[1]))

        if np.random.random() < 0.05:
            # 10% 확률로 무음 신호로 설정
            waveform = torch.zeros_like(waveform)
            label = torch.tensor([0, 0])
            self.silent_samples.append(self.feat[index])  # 무음 신호 샘플의 경로 저장
        else:
            label = torch.tensor(self.label[index])  # label을 Torch Tensor로 변환

        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label

class CustomDataset_test(Dataset):
    def __init__(self, dataframe, sr=32000):
        self.dataframe = dataframe
        self.sr = sr

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        waveform, orig_sr = torchaudio.load(self.dataframe[index])
        # 이미 5초 길이의 데이터를 가정하기 때문에 추가적인 처리는 필요하지 않음
        return waveform
    

data_transform = transforms.Compose([
    # NoiseInjection(noise_levels=(0.0, 0.01))
    RandomCrop(crop_size=16000)   
])

train_dataset = CustomDataset(train_feat, train_labels, CONFIG.SR, transform=data_transform)
val_dataset = CustomDataset(val_feat, val_labels, CONFIG.SR, None)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = torch.zeros_like(x)  # x와 동일한 크기의 0으로 초기화된 텐서
    mixed_y = torch.zeros_like(y)  # y와 동일한 크기의 0으로 초기화된 텐서

    for i in range(batch_size):
        if np.array_equal(y[i].cpu().numpy(), [0, 0]):
            # 무음 샘플은 섞지 않고 그대로 유지
            mixed_x[i] = x[i]
            mixed_y[i] = y[i]
        else:
            # 무음 샘플이 아닌 경우 Mixup 진행
            while np.array_equal(y[index[i]].cpu().numpy(), [0, 0]):
                index[i] = (index[i] + 1) % batch_size  # 무음 샘플을 피하기 위해 인덱스 조정
            mixed_x[i] = lam * x[i] + (1 - lam) * x[index[i]]
            
            y_a, y_b = y[i], y[index[i]]
            
            # 레이블 구분
            if np.array_equal(y_a.cpu().numpy(), [0, 1]) and np.array_equal(y_b.cpu().numpy(), [0, 1]):
                mixed_y[i] = torch.tensor([0, 1])  # real, real
            elif np.array_equal(y_a.cpu().numpy(), [1, 0]) and np.array_equal(y_b.cpu().numpy(), [1, 0]):
                mixed_y[i] = torch.tensor([1, 0])  # fake, fake
            else:
                mixed_y[i] = torch.tensor([1, 1])  # fake, real or real, fake

    return mixed_x, mixed_y, lam


def mixup_criterion(criterion, pred, mixed_y, lam):
    return lam * criterion(pred, mixed_y[:, 0]) + (1 - lam) * criterion(pred, mixed_y[:, 1])

class MixupDataLoader:
    def __init__(self, data_loader, alpha=1.0, fixed_lam=None):
        self.data_loader = data_loader
        self.alpha = alpha
        self.fixed_lam = fixed_lam

    def __iter__(self):
        for x, y in self.data_loader:
            if self.fixed_lam is not None:
                lam = self.fixed_lam
            else:
                lam = np.random.beta(self.alpha, self.alpha)

            mixed_x, mixed_y, _ = mixup_data(x, y, lam)

            yield mixed_x, mixed_y, lam

    def __len__(self):
        return len(self.data_loader)


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

def ResNet18_1D(num_classes=2):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes)


mixup_train_loader = MixupDataLoader(train_loader, alpha=1.0, fixed_lam=0.4)



def train_supervised(model, optimizer, scheduler, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)  # 다중 레이블 분류를 위한 손실 함수

    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS + 1):
        model.train()
        train_loss = []
        
        # Train with mixed-up data
        for features, y_a, lam in tqdm(train_loader):
            features = features.float().to(device)
            y_a = y_a.float().to(device)

            optimizer.zero_grad()
            
            # Forward pass
            logits = model(features)  # logits를 얻기 위해 시그모이드 함수를 적용하지 않음
            loss = criterion(logits, y_a)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
         #Validation
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss: [{_train_loss:.5f}] Val Loss: [{_val_loss:.5f}] Val AUC: [{_val_score:.5f}]')
        
        scheduler.step(_val_score)
        
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            probs = probs.squeeze(1)
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score


model = ResNet18_1D(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
print(model)
infer_model = train_supervised(model, optimizer, scheduler, mixup_train_loader, val_loader, device)

#/home/work/.jun/contest/team_1/jjjjj/test
test = pd.read_csv('/home/work/.jun/contest/team_2/ctest.csv')
test_feat = get_feature(test, False)
test_dataset = CustomDataset_test(test_feat, 32000)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            probs = model(features)
            probs = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions


# 추론 실행
preds = inference(model, test_loader, device)
#print(preds.shape)
# 결과를 제출 파일에 저장
submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:, 1:] = preds#[0]
submit.head(10)

submit.to_csv('./baseline_submit_jiwoo_error_c.csv', index=False)
