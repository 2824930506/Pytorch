import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import configargparse
from tensorboardX import SummaryWriter
import math
# dataset/model/loss function
from dataLoader import data_loader
from selfholo import selfholo
import perceptualloss as perceptualloss
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Command line argument processing
p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--run_id', type=str, default='', help='Experiment name', required=False)
p.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
p.add_argument('--size_of_miniBatches', type=int, default=1, help='Size of minibatch')
p.add_argument('--lr', type=float, default=4e-4, help='learning rate of Holonet weights')

# parse arguments
opt = p.parse_args()
run_id = opt.run_id

# tensorboard setup and file naming
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
writer = SummaryWriter(f'runs/{run_id}_{time_str}')

device = torch.device('cuda')

# Image data for training
train_loader = data_loader(opt)

# Load models #
self_holo = selfholo().to(device)
self_holo.train()  # generator to be trained

# Loss function
loss = perceptualloss.PerceptualLoss(lambda_feat=0.025)
loss = loss.to(device)
mseloss = nn.MSELoss()
mseloss = mseloss.to(device)

# create optimizer
optvars = self_holo.parameters()
optimizer = optim.Adam(optvars, lr=opt.lr)

# Training loop #
for i in range(opt.num_epochs):

    for k, target in enumerate(train_loader):
        # get target image
        amp, depth, mask, ikk = target
        amp, depth, mask = amp.to(device), depth.to(device), mask.to(device)
        source = torch.cat([amp, depth], dim=-3)

        optimizer.zero_grad()

        ik = k + i * len(train_loader)
        output = self_holo(source, ikk)
        print('amp.shape:', amp.shape, 'output.shape:', output.shape)
        loss = mseloss(output, amp**2)
        loss.backward()
        optimizer.step()

        # print and output to tensorboard
        print(f'iteration {ik}: {loss.item()}')

        with torch.no_grad():
            writer.add_scalar('Loss', loss, ik)
            '''
            if ik % 50 == 0:
                writer.add_image('amp', (amp[0, ...]), ik)
                writer.add_image('depth', (depth[0, ...]), ik)
                writer.add_image('output_amp', (output[0, ...]), ik)
                # normalize SLM phase
                writer.add_image('SLM Phase', (holo[0, ...] + math.pi) / (2 * math.pi), ik)
            '''
    # save trained model
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(self_holo.state_dict(), f'checkpoints/{run_id}_{time_str}_{i+1}.pth')

# python ./src/train.py  --run_id=selfholo
