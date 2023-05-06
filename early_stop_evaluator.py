import nni
import torch
import fcntl
import time
import os

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from torch.profiler import profile, record_function, ProfilerActivity


class Evaluator:
    def __init__(self, num_epochs = 3):
        self.num_epochs = num_epochs

    def train_epoch(self, model, device, train_loader, optimizer, epoch):
        loss_fn = torch.nn.CrossEntropyLoss()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            if batch_idx == 5:
                break


    def test_epoch(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        inf_time = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                #torch.cuda.synchronize(device)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
                    with record_function("model_inference"):
                        output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                inf_time += sum([item.cpu_time + item.cuda_time for item in prof.key_averages()])

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        inf_time /= len(test_loader.dataset)

        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset), accuracy))

        return accuracy, inf_time

    def evaluate_model(self, model_cls):
        model = model_cls()

        print("a")
        open_mode = os.O_RDONLY | os.O_CREAT
        fd = os.open('avgep1acc.txt', open_mode)
        pid = os.getpid()
        fcntl.flock(fd, fcntl.LOCK_SH)
        file_info = os.read(fd, os.path.getsize(fd)).rstrip()
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        if len(file_info) == 0:
            ep1_threshold = 0
        else:
            ep1_metric, ep1_models = file_info.split()
            ep1_threshold = float(ep1_metric)/float(ep1_models)
        print("b")

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
        test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)

        for epoch in range(self.num_epochs):
            print(epoch)
            self.train_epoch(model, device, train_loader, optimizer, epoch)
            accuracy, inf_time = self.test_epoch(model, device, test_loader)
            metric = accuracy*(inf_time**-.07)
            if epoch == 0:
                open_mode = os.O_RDWR | os.O_CREAT
                fd = os.open('avgep1acc.txt', open_mode)
                pid = os.getpid()
                fcntl.flock(fd, fcntl.LOCK_SH)
                file_info = os.read(fd, os.path.getsize(fd)).rstrip()
                if len(file_info) == 0:
                    ep1_metric, ep1_models = 0, 0
                else:
                    ep1_metric, ep1_models = file_info.split()
                    ep1_metric, ep1_models = float(ep1_metric), float(ep1_models)
                ep1_metric += metric
                ep1_models += 1
                os.truncate(fd, 0)
                os.lseek(fd, 0, 0)
                os.write(fd, f"{ep1_metric:} {ep1_models}".encode())
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

                if ep1_threshold > metric:
                    break
            nni.report_intermediate_result(accuracy*(inf_time**-.07))


        nni.report_final_result(accuracy*(inf_time**-.07))
