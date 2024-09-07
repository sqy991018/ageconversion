import os
import re
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle




# Returns a function that creates a normalization function
def get_norm_layer(opt):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)


    def add_norm_layer(layer):
        layer = spectral_norm(layer)

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class TFAN_1D(nn.Module):
    def __init__(self, norm_nc, ks=5, label_nc=256, N=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm1d(norm_nc, affine=False)

        self.repeat_N = N

        nhidden = 256

        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv1d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv1d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv1d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        if segmap.dim() == 3:
            segmap = segmap.squeeze(1)
        elif segmap.dim() == 2:
            segmap = segmap.unsqueeze(1)


        segmap = F.interpolate(segmap, size=x.size(2), mode='nearest')


        temp = segmap
        for i in range(self.repeat_N):
            temp = self.mlp_shared(temp)
        actv = temp

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out


class TFAN_2D(nn.Module):
    def __init__(self, norm_nc, ks=5, label_nc=256, N=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.repeat_N = N

        nhidden = label_nc

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        temp = segmap
        for i in range(self.repeat_N):
            temp = self.mlp_shared(temp)
        actv = temp

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


class downSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))

    def forward(self, input):
        return self.convLayer(input) * torch.sigmoid(self.convLayer_gates(input))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 2D Conv Layer
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(5, 15),
                               stride=(1, 1),
                               padding=(2, 7))

        self.conv1_gates = nn.Conv2d(in_channels=1,
                                     out_channels=128,
                                     kernel_size=(5, 15),
                                     stride=1,
                                     padding=(2, 7))

        # 2D Downsample Layer
        self.downSample1 = downSample_Generator(in_channels=128,
                                                out_channels=256,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)

        self.downSample2 = downSample_Generator(in_channels=256,
                                                out_channels=256,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)

        self.conv2dto1dLayer = nn.Conv1d(in_channels=5120,
                                         out_channels=256,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.conv2dto1dLayer_tfan = TFAN_1D(256)

        # Residual Blocks
        self.residualLayer1 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer2 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer3 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer4 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer5 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer6 = ResidualLayer(in_channels=256,
                                            out_channels=512,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        # 1D -> 2D Conv
        self.conv1dto2dLayer = nn.Conv1d(in_channels=256,
                                         out_channels=5120,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.conv1dto2dLayer_tfan = TFAN_1D(5120)

        # UpSample Layer
        self.upSample1 = self.upSample(in_channels=256,
                                       out_channels=1024,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.upSample1_tfan = TFAN_2D(1024 // 4)
        self.glu = GLU()

        self.upSample2 = self.upSample(in_channels=256,
                                       out_channels=512,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)
        self.upSample2_tfan = TFAN_2D(512 // 4)

        self.lastConvLayer = nn.Conv2d(in_channels=128,
                                       out_channels=1,
                                       kernel_size=(5, 15),
                                       stride=(1, 1),
                                       padding=(2, 7))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm1d(
                                           num_features=out_channels,
                                           affine=True),
                                       GLU())

        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor=2))
        return self.convLayer

    def forward(self, input):
        input = input.unsqueeze(1)
        seg_1d = input  # for TFAN module

        conv1 = self.conv1(input) * torch.sigmoid(self.conv1_gates(input))

        # DownSample
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)

        # 2D -> 1D
        n, c, h, w = downsample2.size()
        reshape2dto1d = downsample2.view(n, c * h, w)

        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)

 
        seg_1d = seg_1d.squeeze(1)
        seg_1d = F.interpolate(seg_1d, size=conv2dto1d_layer.size()[2:], mode='nearest')

        if seg_1d.size(1) != conv2dto1d_layer.size(1):
            seg_1d = seg_1d.to(conv2dto1d_layer.device)
            seg_1d = nn.Conv1d(in_channels=seg_1d.size(1), out_channels=conv2dto1d_layer.size(1), kernel_size=1).to(seg_1d.device)(seg_1d)

        conv2dto1d_layer = self.conv2dto1dLayer_tfan(conv2dto1d_layer, seg_1d)

        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)

        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)

        conv1dto2d_layer = self.conv1dto2dLayer_tfan(conv1dto2d_layer, seg_1d)

        n, c, w = conv1dto2d_layer.size()
        reshape1dto2d = conv1dto2d_layer.view(n, c // h, h, w)

        seg_2d = reshape1dto2d

        # UpSample
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_1 = self.upSample1_tfan(upsample_layer_1, seg_2d)
        upsample_layer_1 = self.glu(upsample_layer_1)

        upsample_layer_2 = self.upSample2(upsample_layer_1)
        upsample_layer_2 = self.upSample2_tfan(upsample_layer_2, upsample_layer_1)
        upsample_layer_2 = self.glu(upsample_layer_2)

        output = self.lastConvLayer(upsample_layer_2)

        output = output.squeeze(1)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=128,
                                                  kernel_size=(3, 3),
                                                  stride=(1, 1),
                                                  padding=(1, 1)),
                                        GLU())

        # DownSample Layer
        self.downSample1 = self.downSample(in_channels=128,
                                           out_channels=256,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=1)

        self.downSample2 = self.downSample(in_channels=256,
                                           out_channels=512,
                                           kernel_size=(3, 3),
                                           stride=[2, 2],
                                           padding=1)

        self.downSample3 = self.downSample(in_channels=512,
                                           out_channels=1024,
                                           kernel_size=[3, 3],
                                           stride=[2, 2],
                                           padding=1)

        self.downSample4 = self.downSample(in_channels=1024,
                                           out_channels=1024,
                                           kernel_size=[1, 10],  # [1, 5] for cyclegan-vc2
                                           stride=(1, 1),
                                           padding=(0, 2))

        # Conv Layer
        self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=1024,
                                                       out_channels=1,
                                                       kernel_size=(1, 3),
                                                       stride=[1, 1],
                                                       padding=[0, 1]))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels,
                                                    affine=True),
                                  GLU())
        return convLayer

    def forward(self, input):
        input = input.unsqueeze(1)
        conv_layer_1 = self.convLayer1(input)

        downsample1 = self.downSample1(conv_layer_1)
        downsample2 = self.downSample2(downsample1)
        downsample3 = self.downSample3(downsample2)

        output = torch.sigmoid(self.outputConvLayer(downsample3))
        return output



class LSGANLoss(nn.Module):
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.l1_loss = nn.L1Loss()  # l1 loss
    def forward(self, pred, target):
        return self.l1_loss(pred, target)




#class FeatureMatchingLoss(nn.Module):
#    def __init__(self):
#        super(FeatureMatchingLoss, self).__init__()
#
#    def forward(self, real_features, generated_features):
#        loss = 0
#        for real, gen in zip(real_features, generated_features):
#            loss += nn.functional.l1_loss(gen, real.detach())
#        return loss


# Dataset class
class MelSpecDataset(torch.utils.data.Dataset):
    def __init__(self, data, target_length=64):
        self.data = data
        self.target_length = target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spec = self.data.iloc[idx]['mel_spec']
        mel_spec = torch.tensor(mel_spec)


        if mel_spec.dim() == 3:
            mel_spec = mel_spec.squeeze(0)

        time_steps = mel_spec.shape[-1]
        
        # mel_spec shape [80, 64]
        if time_steps > self.target_length:
            start_idx = np.random.randint(0, time_steps - self.target_length + 1)
            mel_spec = mel_spec[:, start_idx:start_idx + self.target_length]
        else:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, self.target_length - time_steps))

        return mel_spec





def real_labels(size):
    return torch.ones(size, device=device)

def fake_labels(size):
    return torch.zeros(size, device=device)




def load_checkpoint(checkpoint_filename, G_y2m, G_m2y, D_m1, D_y1, optimizer_G, optimizer_D):
    checkpoint = torch.load(checkpoint_filename)
    G_y2m.load_state_dict(checkpoint['G_y2m_state_dict'])
    G_m2y.load_state_dict(checkpoint['G_m2y_state_dict'])
    D_m1.load_state_dict(checkpoint['D_m1_state_dict'])
    D_y1.load_state_dict(checkpoint['D_y1_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    return start_epoch, best_loss


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    
    
def real_labels(size, smoothing=True, device="cuda"):
    if smoothing:
        return torch.ones(size, device=device) * 0.9
    else:
        return torch.ones(size, device=device)

def fake_labels(size, device="cuda"):

    return torch.zeros(size, device=device)






# This is one cyclegan for training y2m and m2y. Please do the same thing for m2o and o2m
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('./kin_dataset/youth_data_16k.pkl', 'rb') as f:
        youth_data = pickle.load(f)

    with open('./kin_dataset/middle_age_data_16k.pkl', 'rb') as f:
        middle_age_data = pickle.load(f)

    with open('./kin_dataset/old_age_data_16k.pkl', 'rb') as f:
        old_data = pickle.load(f)

    youth_dataset = MelSpecDataset(youth_data)
    middle_age_dataset = MelSpecDataset(middle_age_data)
    old_dataset = MelSpecDataset(old_data)

    batch_size = 1
    youth_loader = DataLoader(youth_dataset, batch_size=batch_size, shuffle=True)
    middle_age_loader = DataLoader(middle_age_dataset, batch_size=batch_size, shuffle=True)
    old_loader = DataLoader(old_dataset, batch_size=batch_size, shuffle=True)

    G_y2m = Generator().to(device)
    G_m2y = Generator().to(device)
    D_m1 = Discriminator().to(device)
    D_y1 = Discriminator().to(device)

    criterion_gan = LSGANLoss().to(device)
    cycle_loss_lambda = 10
    identity_loss_lambda = 5

    optimizer_G = optim.Adam(list(G_y2m.parameters()) + list(G_m2y.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(list(D_m1.parameters()) + list(D_y1.parameters()), lr=0.0001, betas=(0.5, 0.999))

    checkpoint_filename = "checkpoint.pth.tar"
    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(checkpoint_filename):
        start_epoch, best_loss = load_checkpoint(checkpoint_filename, G_y2m, G_m2y, D_m1, D_y1, optimizer_G, optimizer_D)
        print(f"Loaded checkpoint from {checkpoint_filename}, starting from epoch {start_epoch} with best loss {best_loss}")
        with open('training_log.txt', 'a') as log_file:
            log_file.write(f"Loaded checkpoint from {checkpoint_filename}, starting from epoch {start_epoch} with best loss {best_loss}")

    def train_model(youth_loader, middle_age_loader, num_epochs=2000, start_epoch=0, best_loss=float('inf'), checkpoint_filename="checkpoint.pth.tar"):
        for epoch in range(start_epoch, num_epochs):

            for youth_mel, middle_mel in zip(youth_loader, middle_age_loader):
                youth_mel = youth_mel.to(device)
                middle_mel = middle_mel.to(device)

                fake_middle = G_y2m(youth_mel)
                cycle_youth = G_m2y(fake_middle)

                fake_youth = G_m2y(middle_mel)
                cycle_middle = G_y2m(fake_youth)

                identity_y = G_m2y(youth_mel)
                identity_m = G_y2m(middle_mel)

                d_fake_youth = D_y1(fake_youth)
                d_fake_middle = D_m1(fake_middle)

                d_fake_cycle_youth = D_y1(cycle_youth)
                d_fake_cycle_middle = D_m1(cycle_middle)

                cycleLoss = torch.mean(torch.abs(youth_mel - cycle_youth)) + torch.mean(torch.abs(middle_mel - cycle_middle))
                identityLoss = torch.mean(torch.abs(youth_mel - identity_y)) + torch.mean(torch.abs(middle_mel - identity_m))

                loss_G_y2m = criterion_gan(d_fake_middle, real_labels(d_fake_middle.size(), device=device))
                loss_G_m2y = criterion_gan(d_fake_youth, real_labels(d_fake_youth.size(), device=device))

                generator_loss = loss_G_y2m + loss_G_m2y + cycle_loss_lambda * cycleLoss + identity_loss_lambda * identityLoss

                optimizer_G.zero_grad()
                generator_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()

                d_real_m = D_m1(middle_mel)
                d_real_y = D_y1(youth_mel)

                loss_D_m_real = criterion_gan(d_real_m, real_labels(d_real_m.size(), device=device))
                loss_D_y_real = criterion_gan(d_real_y, real_labels(d_real_y.size(), device=device))

                loss_D_m_fake = criterion_gan(d_fake_middle.detach(), fake_labels(d_fake_middle.size(), device=device))
                loss_D_y_fake = criterion_gan(d_fake_youth.detach(), fake_labels(d_fake_youth.size(), device=device))

                loss_D_m_cycled = criterion_gan(d_fake_cycle_middle.detach(), fake_labels(d_fake_cycle_middle.size(), device=device))
                loss_D_y_cycled = criterion_gan(d_fake_cycle_youth.detach(), fake_labels(d_fake_cycle_youth.size(), device=device))

                loss_D_m = (loss_D_m_real + loss_D_m_fake) * 0.5 + (loss_D_m_real + loss_D_m_cycled) * 0.5
                loss_D_y = (loss_D_y_real + loss_D_y_fake) * 0.5 + (loss_D_y_real + loss_D_y_cycled) * 0.5

                loss_D = (loss_D_m + loss_D_y) * 0.5
                loss_D.backward()
                optimizer_D.step()

            print(f"Epoch [{epoch}/{num_epochs}] - Generator Loss: {generator_loss.item()}, Discriminator Loss: {loss_D.item()}")

            with open('training_log.txt', 'a') as log_file:
                log_file.write(f"Epoch [{epoch}/{num_epochs}] - Generator Loss: {generator_loss.item()}, Discriminator Loss: {loss_D.item()}\n")

            if generator_loss.item() < best_loss:
                best_loss = generator_loss.item()
                torch.save(G_y2m.state_dict(), 'best_G_y2m.pth')
                torch.save(G_m2y.state_dict(), 'best_G_m2y.pth')
                torch.save(D_m1.state_dict(), 'best_D_m1.pth')
                torch.save(D_y1.state_dict(), 'best_D_y1.pth')
                print(f"Best model saved at epoch {epoch} with loss {best_loss}")
                with open('training_log.txt', 'a') as log_file:
                    log_file.write(f"Best model saved at epoch {epoch} with loss {best_loss}\n")

            save_checkpoint({
                'epoch': epoch + 1,
                'G_y2m_state_dict': G_y2m.state_dict(),
                'G_m2y_state_dict': G_m2y.state_dict(),
                'D_m1_state_dict': D_m1.state_dict(),
                'D_y1_state_dict': D_y1.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_loss': best_loss
            }, filename=checkpoint_filename)
            print(f"Checkpoint saved at epoch {epoch + 1}")

    train_model(youth_loader, middle_age_loader, start_epoch=start_epoch, best_loss=best_loss, checkpoint_filename=checkpoint_filename)
