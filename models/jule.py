import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Argument parser
parser = argparse.ArgumentParser(description='Joint Unsupervised Learning')
parser.add_argument('-dataset', default='UMist', type=str, help='dataset name for evaluation')
parser.add_argument('-eta', default=0.2, type=float, help='unrolling rate for recurrent process')
parser.add_argument('-epoch_rnn', default=1, type=int, help='number of rnn epochs for joint learning')
parser.add_argument('-batchSize', default=100, type=int, help='batch size for training CNN')
parser.add_argument('-learningRate', default=0.01, type=float, help='base learning rate for training CNN')
parser.add_argument('-weightDecay', default=5e-5, type=float, help='weight decay for training CNN')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for training CNN')
parser.add_argument('-gamma_lr', default=0.0001, type=float, help='gamma for inverse learning rate policy')
parser.add_argument('-power_lr', default=0.75, type=float, help='power for inverse learning rate policy')
parser.add_argument('-num_nets', default=1, type=int, help='number of models to train')
parser.add_argument('-epoch_pp', default=20, type=int, help='number of CNN training epochs at each partially unrolled period')
parser.add_argument('-epoch_max', default=1000, type=int, help='number of CNN training epochs in the whole recurrent process')
parser.add_argument('-K_s', default=20, type=int, help='number of neighbors for computing affinity between samples')
parser.add_argument('-K_c', default=5, type=int, help='number of clusters for considering local structure')
parser.add_argument('-gamma_tr', default=1, type=float, help='weight of positive pairs in weighted triplet loss')
parser.add_argument('-margin_tr', default=0.2, type=float, help='margin for weighted triplet loss')
parser.add_argument('-num_nsampling', default=20, type=int, help='number of negative samples for each positive pairs to construct triplet')
parser.add_argument('-use_fast', default=1, type=int, help='whether use fast affinity updating algorithm for acceleration')
parser.add_argument('-updateCNN', default=1, type=int, help='whether update CNN')
parser.add_argument('-centralize_input', default=0, type=int, help='centralize input image data')
parser.add_argument('-centralize_feature', default=0, type=int, help='centralize output feature for clustering')
parser.add_argument('-normalize', default=1, type=int, help='normalize output feature for clustering')
args = parser.parse_args()

# Load dataset
print('==> loading data')
with h5py.File(f'datasets/{args.dataset}/data4torch.h5', 'r') as f:
    trainData_data = torch.tensor(f['data'][:], dtype=torch.float32)
    trainData_label = torch.tensor(f['labels'][:], dtype=torch.float32)

# Centralize training data
if args.centralize_input == 1:
    data_mean = torch.mean(trainData_data, dim=0, keepdim=True)
    trainData_data -= data_mean

testData_data = trainData_data.clone()
testData_label = trainData_label.clone()

# Initialize networks parameters
def net_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Convert labels to label table
def cvt2_table_labels(labels):
    labels_sorted, indices = torch.sort(labels)
    unique_labels, inverse_indices = torch.unique(labels_sorted, return_inverse=True)
    nclasses = unique_labels.size(0)
    labels_from_one = inverse_indices + 1
    labels_tb = [[] for _ in range(nclasses)]
    for i in range(labels.size(0)):
        labels_tb[labels_from_one[i].item() - 1].append(i)
    return labels_tb

# CNN model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define your model layers here
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize networks
num_networks = args.num_nets
network_table = []
parameters_table = []
optimizers = []
label_gt_table_table = []
label_gt_tensor_table = []
label_pre_table_table = []
label_pre_tensor_table = []
target_nclusters_table = torch.zeros(num_networks, dtype=torch.long)
epoch_reset_labels = torch.zeros(num_networks, dtype=torch.long)

for i in range(num_networks):
    model = SimpleCNN().cuda()
    model.apply(net_init)
    network_table.append(model)
    parameters_table.append(list(model.parameters()))
    optimizers.append(optim.SGD(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay, momentum=args.momentum))
    label_gt_tensor_table.append(testData_label)
    label_gt_table_table.append(cvt2_table_labels(testData_label))
    target_nclusters_table[i] = len(label_gt_table_table[i])

# Set criterion
criterion_triplet = nn.TripletMarginLoss(margin=args.margin_tr, p=2).cuda()

# Extract features for images from CNN
def extract_features(model, data):
    model.eval()
    with torch.no_grad():
        features = model(data.cuda()).cpu()
    return features

# Convert table labels to tensor labels
def cvt2_tensor_labels(labels, ind_s, ind_e):
    label_tensor = torch.zeros(ind_e - ind_s + 1, 1)
    for i, label_group in enumerate(labels):
        for j in label_group:
            label_tensor[j, 0] = i + 1
    return label_tensor

# Define affinity functions
def compute_affinity(features, k_s):
    nbrs = NearestNeighbors(n_neighbors=k_s, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)
    W = np.exp(-distances ** 2)
    return distances, indices, W

def update_labels(features, label_pre, target_clusters, iter):
    distances, indices, W = compute_affinity(features.numpy(), args.K_s)
    W = csr_matrix(W)
    if iter == 0:
        print("initialize clusters...")
        n_components, labels = connected_components(csgraph=W, directed=False, return_labels=True)
        label_pre = [np.where(labels == i)[0].tolist() for i in range(n_components)]
        return label_pre
    # More sophisticated clustering updates can be added here
    return label_pre

# Merging labels during training
def merge_labels():
    for i, model in enumerate(network_table):
        if epoch_reset_labels[i] == 0 or args.updateCNN == 0:
            features = trainData_data.view(trainData_data.size(0), -1).float()
        else:
            features = extract_features(model, trainData_data)
        
        # Centralize
        if args.centralize_feature == 1:
            features -= features.mean(dim=0, keepdim=True)

        # Normalize
        if args.normalize == 1:
            features = features / features.norm(p=2, dim=1, keepdim=True)
        
        label_pre_table_table[i] = update_labels(features, label_pre_table_table[i], target_nclusters_table[i], epoch_reset_labels[i])
        epoch_reset_labels[i] += 1
        label_pre_tensor_table[i] = cvt2_tensor_labels(label_pre_table_table[i], 1, trainData_data.size(0))

# Define organize_samples function
def organize_samples(X, y):
    num_s = X.size(0)
    y_table = cvt2_table_labels(y)
    nclusters = len(y_table)
    if nclusters == 1:
        return
    num_neg_sampling = args.num_nsampling
    if nclusters <= args.num_nsampling:
        num_neg_sampling = nclusters - 1
    num_triplet = 0
    for i in range(nclusters):
        if len(y_table[i]) > 1:
            num_triplet += len(y_table[i]) * (len(y_table[i]) - 1) * num_neg_sampling // 2
    if num_triplet == 0:
        return
    A = torch.zeros((num_triplet, X.size(1)), dtype=torch.float32).cuda()
    B = torch.zeros((num_triplet, X.size(1)), dtype=torch.float32).cuda()
    C = torch.zeros((num_triplet, X.size(1)), dtype=torch.float32).cuda()
    A_ind = torch.zeros(num_triplet, dtype=torch.long).cuda()
    B_ind = torch.zeros(num_triplet, dtype=torch.long).cuda()
    C_ind = torch.zeros(num_triplet, dtype=torch.long).cuda()
    id_triplet = 0
    for i in range(nclusters):
        if len(y_table[i]) > 1:
            for m in range(len(y_table[i])):
                for n in range(m + 1, len(y_table[i])):
                    is_choosed = torch.zeros(num_s, dtype=torch.int16).cuda()
                    while True:
                        id_s = torch.randint(0, num_s, (1,)).item()
                        id_t = y_table[i][m]
                        if is_choosed[id_s] == 0 and y[id_s] != y[id_t]:
                            A_ind[id_triplet] = y_table[i][m]
                            B_ind[id_triplet] = y_table[i][n]
                            C_ind[id_triplet] = id_s
                            is_choosed[id_s] = 1
                            id_triplet += 1
                        if torch.sum(is_choosed) == num_neg_sampling:
                            break
    A.index_copy_(0, torch.arange(num_triplet).cuda(), X.index_select(0, A_ind))
    B.index_copy_(0, torch.arange(num_triplet).cuda(), X.index_select(0, B_ind))
    C.index_copy_(0, torch.arange(num_triplet).cuda(), X.index_select(0, C_ind))
    return [A, B, C], [A_ind, B_ind, C_ind]

# Define updateCNN function
def update_cnn():
    for model in network_table:
        model.train()
    epoch = 0
    for epoch in range(args.epoch_max):
        print(f'==> online epoch #{epoch + 1} [batchSize = {args.batchSize}] [learningRate = {args.learningRate}]')
        for batch in DataLoader(TensorDataset(trainData_data, trainData_label), batch_size=args.batchSize, shuffle=True):
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()
            for model, optimizer in zip(network_table, optimizers):
                optimizer.zero_grad()
                outputs = model(inputs)
                triplets, triplets_ind = organize_samples(outputs, targets.float())
                if triplets:
                    loss = criterion_triplet(*triplets)
                    loss.backward()
                    optimizer.step()
                    print(f'Loss: {loss.item()}')
        epoch += 1

# Evaluate performance
def evaluate_performance():
    for model in network_table:
        model.eval()
    print('==> testing')
    for labels_gt, labels_pre in zip(label_gt_table_table, label_pre_table_table):
        nmi = normalized_mutual_info_score(
            torch.cat(labels_gt).numpy(),
            torch.cat([torch.tensor(lbl) for lbl in labels_pre]).numpy()
        )
        print(f'NMI: {nmi}')

# Check if all models are finished
def is_all_finished():
    flag = True
    for labels in label_pre_table_table:
        if len(labels) > target_nclusters_table[i]:
            flag = False
    return flag

# Main training loop
for n in range(args.epoch_rnn):
    for i in range(args.epoch_max):
        if i % args.epoch_pp == 0:
            merge_labels()
            evaluate_performance()
            if is_all_finished():
                break
        if args.updateCNN == 1:
            update_cnn()
    epoch_reset_labels.zero_()
    while True:
        merge_labels()
        evaluate_performance()
        if is_all_finished():
            break