"""
Training script adapted from demo in
https://github.com/gpleiss/efficient_densenet_pytorch
"""

import fire
import os
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from models import DenseNetEfficientMulti


class Meter():
    """
    A little helper class which keeps track of statistics during an epoch.
    """
    def __init__(self, name, cum=False):
        """
        name (str or iterable): name of values for the meter
            If an iterable of size n, updates require a n-Tensor
        cum (bool): is this meter for a cumulative value (e.g. time)
            or for an averaged value (e.g. loss)? - default False
        """
        self.cum = cum
        if type(name) == str:
            name = (name,)
        self.name = name

        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0

    def update(self, data, n=1):
        """
        Update the meter
        data (Variable, Tensor, or float): update value for the meter
            Size of data should match size of ``name'' in the initialized args
        """
        self._count = self._count + n
        if isinstance(data, torch.autograd.Variable):
            self._last_value.copy_(data.data)
        elif isinstance(data, torch.Tensor):
            self._last_value.copy_(data)
        else:
            self._last_value.fill_(data)
        self._total.add_(self._last_value)

    def value(self):
        """
        Returns the value of the meter
        """
        if self.cum:
            return self._total
        else:
            return self._total / self._count

    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])


def _set_lr(optimizer, epoch, n_epochs, lr):
    lr = lr
    if float(epoch) / n_epochs > 0.75:
        lr = lr * 0.01
    elif float(epoch) / n_epochs > 0.5:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])


def run_epoch(loader, model, criterion, optimizer, epoch=0, n_epochs=0, train=True):
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)
    error_meter = Meter(name='Error', cum=False)

    if train:
        model.train()
        print('Training')
    else:
        model.eval()
        print('Evaluating')

    end = time.time()
    for i, (input, target) in enumerate(loader):
        if train:
            model.zero_grad()
            optimizer.zero_grad()

        # Forward pass
        input_var = Variable(input, volatile=(not train)).cuda(async=True)
        target_var = Variable(target, volatile=(not train), requires_grad=False).cuda(async=True)
        output_var = model(input_var)
        loss = criterion(output_var, target_var)

        # Backward pass
        if train:
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

        # Accounting
        _, predictions_var = torch.topk(output_var, 1)
        error = 1 - torch.eq(predictions_var, target_var).float().mean()
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        error_meter.update(error)
        print('  '.join([
            '%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
                epoch, n_epochs, i + 1, len(loader)),
            str(time_meter),
            str(loss_meter),
            str(error_meter),
        ]))

    return time_meter.value(), loss_meter.value(), error_meter.value()


def train(data, save, valid_size=5000, seed=None,
          depth=40, growth_rate=12, n_epochs=300, batch_size=64,
          lr=0.1, wd=0.0001, momentum=0.9):
    """
    A function to train a DenseNet-BC on CIFAR-100.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)

        valid_size (int) - size of validation set
        seed (int) - manually set the random seed (default None)

        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        lr (float) - initial learning rate
        wd (float) - weight decay
        momentum (float) - momentum
    """

    if seed is not None:
        torch.manual_seed(seed)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # Split training into train and validation - needed for calibration
    #
    # IMPORTANT! We need to use the same validation set for temperature
    # scaling, so we're going to save the indices for later
    data_root = os.path.join(data, 'cifar100')
    train_set = tv.datasets.CIFAR100(data_root, train=True, transform=train_transforms, download=True)
    valid_set = tv.datasets.CIFAR100(data_root, train=True, transform=test_transforms, download=False)
    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices) - valid_size]
    valid_indices = indices[len(indices) - valid_size:] if valid_size else None

    # Make dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices))
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))

    # Make model, criterion, and optimizer
    model = DenseNetEfficientMulti(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=100
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

    # Wrap model if multiple gpus
    if torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    else:
        model_wrapper = model.cuda()

    # Train model
    best_error = 1
    for epoch in range(1, n_epochs + 1):
        _set_lr(optimizer, epoch, n_epochs, lr)
        run_epoch(
            loader=train_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=True,
        )
        valid_results = run_epoch(
            loader=valid_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=False,
        )

        # Determine if model is the best
        _, _, valid_error = valid_results
        if valid_error[0] < best_error:
            best_error = valid_error[0]
            print('New best error: %.4f' % best_error)

            # When we save the model, we're also going to
            # include the validation indices
            to_save = {
                model: model.state_dict(),
                valid_indices: valid_indices,
            }
            torch.save(to_save, os.path.join(save, 'model_and_validation_indices.pth'))

    print('Done!')


if __name__ == '__main__':
    """
    Train a 40-layer DenseNet-BC on CIFAR-100

    Args:
        --data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        --save (str) - path to save the model to (default /tmp)

        --valid_size (int) - size of validation set
        --seed (int) - manually set the random seed (default None)
    """
    fire.Fire(train)
