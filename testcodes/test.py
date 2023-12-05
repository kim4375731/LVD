import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        print(f'arange: {arange.shape}')
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        print(f'base_grid: {base_grid.shape}')
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        print(f'base_grid2: {base_grid.shape}')

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RandomShiftsAug3D(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, d, h, w = x.size()
        assert h == w and h == d
        padding = tuple([self.pad] * 6)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange[None].repeat(h, 1)[None].repeat(h, 1, 1).unsqueeze(3)
        base_grid = torch.cat([arange, arange.transpose(1,2), arange.transpose(0, 2)], dim=3)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 1, 3),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
    
def main2d():
    side = 4
    pad = 0
    inp = torch.arange(side**2).float().reshape(1,1,side,side) +1
    rs = RandomShiftsAug(pad)
    out = rs(inp)
    print(f'input : \n{inp.int()}')
    print(f'output : \n{out.int()}')
    print(f'inputshaep : {inp.shape}')
    print(f'outputshaep : {out.shape}')

def main3d():
    side = 4
    pad = 1
    inp = torch.arange(side**3).float().reshape(1,1,side,side,side) +1
    rs = RandomShiftsAug3D(pad)
    out = rs(inp)
    print(f'input : \n{inp.int()}')
    print(f'output : \n{out.int()}')
    print(f'inputshaep : {inp.shape}')
    print(f'outputshaep : {out.shape}')


if __name__ == "__main__":
    main3d()