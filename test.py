from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/test")
writer.add_scalar('Accuracy/train', 0.6, 1)
writer.add_scalar('Accuracy/train', 0.6, 2)
writer.add_scalar('Accuracy/train', 0.7, 3)
writer.close()