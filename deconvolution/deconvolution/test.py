from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

def load_event(event_path):
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    loss_train = ea.scalars.Items('Loss/train')
    loss_test = ea.scalars.Items('Loss/test')
    acc_train = ea.scalars.Items('Accuracy/train')
    acc_test = ea.scalars.Items('Accuracy/test')
    return loss_train, loss_test, acc_train, acc_test

_, bn_loss, _, bn_acc = load_event('events.out.tfevents.1619986674.pc-74-128.customer.ask4.lan.77899.0')
bn_loss = [l[2] for l in bn_loss]
bn_acc = [a[2] for a in bn_acc]

_, deconv_loss, _, deconv_acc = load_event('events.out.tfevents.1620016040.pc-74-128.customer.ask4.lan.80335.0')
deconv_loss = [l[2] for l in deconv_loss]
deconv_acc = [a[2] for a in deconv_acc]

plt.figure()
sns.lineplot(x=range(1, 1 + len(bn_loss)), y=bn_loss, label='bn')
sns.lineplot(x=range(1, 1 + len(deconv_loss)), y=deconv_loss, label='deconv')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.tight_layout()
plt.savefig('vgg11_loss_ep20.png')

plt.figure()
sns.lineplot(x=range(1, 1 + len(bn_acc)), y=bn_acc, label='bn')
sns.lineplot(x=range(1, 1 + len(deconv_acc)), y=deconv_acc, label='deconv')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.tight_layout()
plt.savefig('vgg11_acc_ep20.png')
plt.show()
