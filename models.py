import torch.nn as nn
import torch


class SUREfcNoisyMNIST(nn.Module):
    def __init__(self):
        super(SUREfcNoisyMNIST, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 784))
        self.decoder1 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 784))


    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1


class SUREfcCaltech(nn.Module):
    def __init__(self):
        super(SUREfcCaltech, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(1984, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )
        self.decoder0 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 1984))
        self.decoder1 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 512))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1


class SUREfcScene(nn.Module):  # 20, 59
    def __init__(self):
        super(SUREfcScene, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(20, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(59, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 20))
        self.decoder1 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 59))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1


class SUREfcReuters(nn.Module):
    def __init__(self):
        super(SUREfcReuters, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )
        self.decoder0 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 10))
        self.decoder1 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 10))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1


class SUREfcMNISTUSPS(nn.Module):
    def __init__(self):
        super(SUREfcMNISTUSPS, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 784))
        self.decoder1 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 256))


    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1


class SUREfcDeepCaltech(nn.Module):
    def __init__(self):
        super(SUREfcDeepCaltech, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )
        self.decoder0 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 4096))
        self.decoder1 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 4096))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder0(x1.view(x1.size()[0], -1))
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder0(union)
        return h0, h1, z0, z1


class SUREfcDeepAnimal(nn.Module):
    def __init__(self):
        super(SUREfcDeepAnimal, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )
        self.decoder0 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 4096))
        self.decoder1 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 4096))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder0(x1.view(x1.size()[0], -1))
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder0(union)
        return h0, h1, z0, z1
