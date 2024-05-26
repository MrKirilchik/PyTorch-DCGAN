import os
import json
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
from gui import get_user_input, training_completed

trs = {}
for file in os.listdir():
    if file.endswith(".lang"):
        with open(file, 'r', encoding='utf-8') as f:
            trs = json.load(f)

dataset_dir="dataset"
output_dir = "output"
model_dir = "model_save.pth"

n_epochs, batch_size, image_size_x, image_size_y, device_choice, start_choice = get_user_input()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),  
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),  
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),  
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),  
            nn.ConvTranspose2d(64, 4, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

transform = transforms.Compose([
    transforms.Resize((image_size_x, image_size_y)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
])

def rgba_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

dataset = ImageFolder(root=dataset_dir, transform=transform, loader=rgba_loader)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() and device_choice == "gpu" else "cpu")

z_dim = 100
gen = Generator(z_dim).to(device)
disc = Discriminator().to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)
disc_opt = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)

criterion = nn.BCELoss()

def train_model(start_epoch=0):
    for epoch in range(start_epoch, n_epochs):
        for real, _ in dataloader:
            real = real.to(device)
            noise = torch.randn(real.shape[0], z_dim, 1, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            disc.zero_grad()
            loss_disc.backward()
            disc_opt.step()

            output = disc(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            gen_opt.step()

        print(f"{trs['epoch']} {epoch+1} {trs['out_of']} {n_epochs}, {trs['gen_loss']}: {loss_gen.item()}, {trs['disc_loss']}: {loss_disc.item()}")

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'gen_state_dict': gen.state_dict(),
                'gen_opt_state_dict': gen_opt.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'disc_opt_state_dict': disc_opt.state_dict(),
            }, model_dir)
            save_image(fake, f"{output_dir}/output_{epoch}.png", nrow=3)

    training_completed()

if os.path.isfile(model_dir) and start_choice == "resume":
    print(trs["model_found"])
    checkpoint = torch.load(model_dir)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])
    start_epoch = checkpoint['epoch']
    train_model(start_epoch)
else:
    print(trs["model_not_found"])
    train_model()
