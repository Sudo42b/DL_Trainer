import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet
import matplotlib.pyplot as plt
from utils.lr_scheduler import CosineAnnealingWarmupRestarts

def plot(lr_list):
    f = plt.figure()

    plt.plot(lr_list)
    plt.show()


epochs = 200
iterations = 100
model = AlexNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-4, last_epoch=-1)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=1e-4, last_epoch=-1)
scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=iterations,
                                          cycle_mult=0.5,
                                          max_lr=0.1,
                                          min_lr=0.0,
                                          warmup_steps=1,
                                          gamma=0.5)
# this zero gradient update is needed to avoid a warning message, issue #8.
optimizer.zero_grad()

lr_list = list()
for epoch in range(epochs):
    optimizer.step()
    # 참고 : 현재주기의주기 수와 반복 수를 입력해야합니다.
    print(epoch // iterations, epoch % iterations)
    # scheduler.step(epoch // iterations + epoch % iterations)
    scheduler.step()

    print('{} - {}'.format(epoch, scheduler.get_last_lr()))
    lr_list.append(scheduler.get_last_lr()[0])

plot(lr_list)