import torch
import numpy as np
import math

def detector_region(x):
    # 最后的检测层
    # 分为10块进行检测
    return torch.cat((
        x[:, 46 : 66, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 78 : 98].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 109 : 129].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)


class DiffractiveLayer(torch.nn.Module):
    # 初始化一个衍射层
    def __init__(self):
        super(DiffractiveLayer, self).__init__()
        self.size = 200                         # 200 * 200 neurons in one layer 一层的单元数量
        self.distance = 0.03                    # distance bewteen two layers (3cm) 每两个层之间的距离，单位为m
        self.ll = 0.08                          # layer length (8cm)    层的大小0.08*0.08的正方形
        self.wl = 3e8 / 0.4e12                  # wave length            激光波长
        self.fi = 1 / self.ll                   # frequency interval
        self.wn = 2 * math.pi / self.wl       # wave number
        # self.phi (200, 200)
        self.phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size), dtype=np.complex64)
        # h (200, 200)
        # 传播公式
        h = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))
        # self.h (200, 200, 2)
        self.h = torch.nn.Parameter(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1), requires_grad=False)

    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        temp = torch.fft(waves, signal_ndim=2)
        k_pace_real = self.h[..., 0] * temp[..., 0] - self.h[..., 1] * temp[..., 1]
        k_space_imag = self.h[..., 0] * temp[..., 1] + self.h[..., 1] * temp[..., 0]
        k_space = torch.stack((k_pace_real, k_space_imag), dim=-1)
        # angular_spectrum (batch, 200, 200, 2)
        angular_spectrum = torch.ifft(k_space, signal_ndim=2)
        return angular_spectrum


class Net(torch.nn.Module):
    """
    只训练相位
    phase only modulation
    """
    # 共五层
    def __init__(self, num_layers=5):

        super(Net, self).__init__()
        # self.phase (200, 200)
        self.phase = [torch.nn.Parameter(torch.from_numpy(
            2 * np.pi * np.random.random(size=(200, 200)).astype('float32'))) for _ in range(num_layers)]
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])
        self.diffractive_layers = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])
        self.last_diffractive_layer = DiffractiveLayer()
        self.sofmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # x (batch, 200, 200, 2)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(x)
            exp_j_phase = torch.stack((torch.cos(self.phase[index]), torch.sin(self.phase[index])), dim=-1)
            x_real = temp[..., 0] * exp_j_phase[..., 0] - temp[..., 1] * exp_j_phase[..., 1]
            x_imag = temp[..., 0] * exp_j_phase[..., 1] + temp[..., 1] * exp_j_phase[..., 0]
            x = torch.stack((x_real, x_imag), dim=-1)
        x = self.last_diffractive_layer(x)
        # x_abs (batch, 200, 200)
        x_abs = torch.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1])
        output = self.sofmax(detector_region(x_abs))
        return output


if __name__ == '__main__':
    print(Net())
    net = Net()
    net.eval()
    mnist_Data = 1
    out = net(mnist_Data)

    data = torch.randn(size=(1,1,32,32))
    out = net(data)
    print("out.shape",out.shape)
