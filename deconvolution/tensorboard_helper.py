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

root_dir = Path('/Users/guliqi/Downloads/log')
net = 'vgg16'
ep = 100
losses = {}
accuracies = {}
for path in root_dir.glob('*' + net + ',ep.' + str(ep) + '*deconv.1*'):
    for event_path in path.glob('events*'):
        loss_train, loss_test, acc_train, acc_test = load_event(str(event_path))
        loss_test = [l[2] for l in loss_test]
        acc_test = [a[2] for a in acc_test]
        losses['deconv'] = loss_test
        accuracies['deconv'] = acc_test
for path in root_dir.glob('*' + net + ',ep.' + str(ep) + '*deconv.0*'):
    for event_path in path.glob('events*'):
        loss_train, loss_test, acc_train, acc_test = load_event(str(event_path))
        loss_test = [l[2] for l in loss_test]
        acc_test = [a[2] for a in acc_test]
        losses['bn'] = loss_test
        accuracies['bn'] = acc_test

plt.figure()
sns.lineplot(x=np.arange(1, 1 + len(losses['deconv'])),
             y=losses['deconv'], label='deconv')
sns.lineplot(x=np.arange(1, 1 + len(losses['bn'])),
             y=losses['bn'], label='bn')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.tight_layout()
# plt.savefig(net + '_loss_ep' + str(ep) + '.png')

plt.figure()
sns.lineplot(x=np.arange(1, 1 + len(accuracies['deconv'])),
             y=accuracies['deconv'], label='deconv')
sns.lineplot(x=np.arange(1, 1 + len(accuracies['bn'])),
             y=accuracies['bn'], label='bn')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.tight_layout()
# plt.savefig(net + '_acc_ep' + str(ep) + '.png')

plt.show()
