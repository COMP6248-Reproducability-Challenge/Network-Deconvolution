from models import *
from datasets import load_dataset
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import *
import shutil
import math
import os

from arg_parser import parse_args
from tqdm.autonotebook import tqdm


args = parse_args()

n_classes, train_set, test_set = load_dataset(args)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if args.arch == 'vgg16':
    model = Vgg16(n_classes, deconv=args.deconv, delinear=args.delinear).to(device)
elif args.arch == 'resnet18':
    model = ResNet18(n_classes, deconv=args.deconv, delinear=args.delinear).to(device)
else:
    raise Exception

parameters = filter(lambda p: p.requires_grad, model.parameters())
if args.optimizer == 'SGD':
    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=0.001)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(parameters, lr=0.001, weight_decay=0.001)
else:
    raise Exception
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, math.ceil(len(train_set) / args.batch_size) * 20)

criterion = nn.CrossEntropyLoss().to(device)

ckpt_dir = save_path_formatter(args)
writer = SummaryWriter(ckpt_dir, flush_secs=30)
train_logger = Logger(os.path.join(ckpt_dir, 'train.log'),
                      ['epoch', 'loss', 'top1', 'top5'])
test_logger = Logger(os.path.join(ckpt_dir, 'test.log'),
                     ['epoch', 'loss', 'top1', 'top5'])

train_top1 = AverageMeter()
train_top5 = AverageMeter()
train_loss = AverageMeter()
test_top1 = AverageMeter()
test_top5 = AverageMeter()
test_loss = AverageMeter()

best_acc = 0
for epoch in range(args.epochs):
    tk0 = tqdm(train_loader)
    tk0.set_description('Epoch {}/{}'.format(epoch + 1, args.epochs))
    is_best_model = False

    model.train()
    train_top1.reset()
    train_top5.reset()
    train_loss.reset()
    for batch_idx, (inputs, targets) in enumerate(tk0):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if epoch < 15:
            scheduler.step()

        top1, top5 = accuracy(outputs.data, targets, topk=(1, 5))
        train_top1.update(top1, inputs.size(0))
        train_top5.update(top5, inputs.size(0))
        train_loss.update(loss.item(), inputs.size(0))

        tk0.set_postfix(running_loss=loss.item(), train_loss=train_loss.avg, top1=train_top1.avg, top5=train_top5.avg,
                        lr=scheduler.get_last_lr()[0])
    tk0.close()

    if args.log:
        train_logger.log({'epoch': epoch,
                          'loss': train_loss.avg,
                          'top1': train_top1.avg,
                          'top5': train_top5.avg})

    model.eval()
    test_top1.reset()
    test_top5.reset()
    test_loss.reset()
    with torch.no_grad():
        tk1 = tqdm(test_loader)
        tk1.set_description('Epoch {}/{}'.format(epoch + 1, args.epochs))

        for batch_idx, (inputs, targets) in enumerate(tk1):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            top1, top5 = accuracy(outputs.data, targets, topk=(1, 5))
            test_top1.update(top1, inputs.size(0))
            test_top5.update(top5, inputs.size(0))
            test_loss.update(loss.item(), inputs.size(0))

            tk1.set_postfix(running_loss=loss.item(), test_loss=test_loss.avg, top1=test_top1.avg, top5=test_top5.avg)

        tk1.close()

    if args.log:
        test_logger.log({'epoch': epoch,
                         'loss': test_loss.avg,
                         'top1': test_top1.avg,
                         'top5': test_top5.avg})

    # write to tensorboard
    if args.tensorboard:
        writer.add_scalar('Loss/train', train_loss.avg, epoch + 1)
        writer.add_scalar('Loss/test', test_loss.avg, epoch + 1)
        writer.add_scalar('Accuracy/train', train_top1.avg, epoch + 1)
        writer.add_scalar('Accuracy/test', test_top1.avg, epoch + 1)

    # save the model
    if args.save_model:
        if test_top1.avg > best_acc:
            best_acc = test_top1.avg
            is_best_model = True

        states = {
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }
        save_file_path = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
        torch.save(states, save_file_path)
        if is_best_model:
            shutil.copyfile(save_file_path, os.path.join(ckpt_dir, 'model_best.pth.tar'))