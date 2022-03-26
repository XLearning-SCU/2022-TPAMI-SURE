import torch
import numpy as np


def pvp_infer(model, device, all_loader):
    model.eval()
    align_out0 = []
    align_out1 = []
    class_labels_cluster = []
    len_alldata = len(all_loader.dataset)
    align_labels = torch.zeros(len_alldata)
    with torch.no_grad():
        for batch_idx, (x0, x1, labels, class_labels0, class_labels1) in enumerate(all_loader):
            test_num = len(labels)

            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
            x0 = x0.view(x0.size()[0], -1)
            x1 = x1.view(x1.size()[0], -1)
            h0, h1, _, _ = model(x0, x1)

            C = euclidean_dist(h0, h1)
            for i in range(test_num):
                idx = torch.argsort(C[i, :])
                C[:, idx[0]] = float("inf")
                align_out0.append((h0[i, :].cpu()).numpy())
                align_out1.append((h1[idx[0], :].cpu()).numpy())
                if class_labels0[i] == class_labels1[idx[0]]:
                    align_labels[1024 * batch_idx + i] = 1

            class_labels_cluster.extend(class_labels0.numpy())

    count = torch.sum(align_labels)
    inference_acc = count.item() / len_alldata

    return np.array(align_out0), np.array(align_out1), np.array(class_labels_cluster), inference_acc


def pdp_infer(model, device, all_loader):
    model.eval()
    recover_out0 = []  # view 0 for learned selecting filling
    recover_out1 = []
    class_labels = []
    with torch.no_grad():
        k = 3
        for batch_idx, (x0, x1, labels, class_labels0, class_labels1, mask) in enumerate(all_loader):
            test_num = len(labels)
            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
            x0 = x0.view(x0.size()[0], -1)
            x1 = x1.view(x1.size()[0], -1)
            class_labels.extend((labels.cpu()).numpy())
            h0, h1, _, _ = model(x0, x1)

            if mask.sum() == test_num:  # complete
                continue

            fill_num = k
            C = euclidean_dist(h0, h1)
            row_idx = C.argsort()
            col_idx = (C.t()).argsort()
            # Mij denotes the flag of i-th sample in view 0 and j-th sample in view 1
            M = torch.logical_and((mask[:, 0].repeat(test_num, 1)).t(), mask[:, 1].repeat(test_num, 1))

            for i in range(test_num):
                idx0 = col_idx[i, :][M[col_idx[i, :], i]]  # idx for view 0 to sort and find the non-missing neighbors
                idx1 = row_idx[i, :][M[i, row_idx[i, :]]]  # idx for view 1 to sort and find the non-missing neighbors
                if len(idx1) != 0 and len(idx0) == 0:  # i-th sample in view 1 is missing
                    # weight = torch.softmax(h1[idx1[0:fill_num], :], dim=0)
                    # avg_fill = (weight * h1[idx1[0:fill_num], :]).sum(dim=0)
                    avg_fill = h1[idx1[0:fill_num], :].sum(dim=0) / fill_num
                    recover_out0.append((h0[i, :].cpu()).numpy())
                    recover_out1.append((avg_fill.cpu()).numpy())  # missing
                    # missing_cnt += 1
                elif len(idx0) != 0 and len(idx1) == 0:  # i-th sample in view 0 is missing
                    # weight = torch.softmax(h0[idx0[0:fill_num], :], dim=0)
                    # avg_fill = (weight * h0[idx0[0:fill_num], :]).sum(dim=0)
                    avg_fill = h0[idx0[0:fill_num], :].sum(dim=0) / fill_num
                    recover_out0.append((avg_fill.cpu()).numpy())  # missing
                    recover_out1.append((h1[i, :].cpu()).numpy())
                    # missing_cnt += 1
                elif len(idx0) != 0 and len(idx1) != 0:  # complete
                    recover_out0.append((h0[i, :].cpu()).numpy())
                    recover_out1.append((h1[i, :].cpu()).numpy())
                else:
                    raise Exception('error')

    return np.array(recover_out0), np.array(recover_out1), np.array(class_labels)


def both_infer(model, device, all_loader, setting):
    model.eval()
    align_out0 = []
    align_out1 = []
    class_labels = []
    len_alldata = len(all_loader.dataset)
    with torch.no_grad():
        cnt = 0
        k = 3
        missing_cnt = 0
        for batch_idx, (x0, x1, labels, class_labels0, class_labels1, mask) in enumerate(all_loader):
            test_num = len(labels)
            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
            x0 = x0.view(x0.size()[0], -1)
            x1 = x1.view(x1.size()[0], -1)
            class_labels.extend((labels.cpu()).numpy())
            h0, h1, _, _ = model(x0, x1)
            recover_out0 = (torch.empty_like(h0)).to(device)
            recover_out1 = (torch.empty_like(h1)).to(device)

            fill_num = k
            C = euclidean_dist(h0, h1)
            row_idx = C.argsort()
            col_idx = (C.t()).argsort()
            # Mij denotes the flag of i-th sample in view 0 and j-th sample in view 1
            M = torch.logical_and((mask[:, 0].repeat(test_num, 1)).t(), mask[:, 1].repeat(test_num, 1))
            for i in range(test_num):
                idx0 = col_idx[i, :][M[col_idx[i, :], i]]  # idx for view 0 to sort and find the non-missing neighbors
                idx1 = row_idx[i, :][M[i, row_idx[i, :]]]  # idx for view 1 to sort and find the non-missing neighbors
                if len(idx1) != 0 and len(idx0) == 0:  # i-th sample in view 1 is missing
                    avg_fill = h1[idx1[0:fill_num], :].sum(dim=0) / fill_num
                    cnt += (class_labels1[idx1[0:fill_num]] == class_labels1[i]).sum()
                    missing_cnt += 1
                    recover_out0[i, :] = h0[i, :]
                    recover_out1[i, :] = avg_fill  # missing
                elif len(idx0) != 0 and len(idx1) == 0:
                    avg_fill = h0[idx0[0:fill_num], :].sum(dim=0) / fill_num
                    cnt += (class_labels0[idx0[0:fill_num]] == class_labels0[i]).sum()
                    missing_cnt += 1
                    recover_out0[i, :] = avg_fill  # missing
                    recover_out1[i, :] = h1[i, :]
                elif len(idx0) != 0 and len(idx1) != 0:
                    recover_out0[i, :] = h0[i, :]
                    recover_out1[i, :] = h1[i, :]
                else:
                    raise Exception('error')
            if setting == 1:
                align_out0.extend((recover_out0.cpu()).numpy())
                align_out1.extend((recover_out1.cpu()).numpy())
                continue
            C = euclidean_dist(recover_out0, recover_out1)
            for i in range(test_num):
                idx = torch.argsort(C[i, :])
                C[:, idx[0]] = float("inf")
                align_out0.append((recover_out0[i, :].cpu()).numpy())
                align_out1.append((recover_out1[idx[0], :].cpu()).numpy())

    return np.array(align_out0), np.array(align_out1), np.array(class_labels)


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
