import nni
import torch
import fcntl
import time
import os

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from torch.profiler import profile, record_function, ProfilerActivity

class EarlyTerminationEvaluator:
    def __init__(self, num_epochs = 3, cull_ratio=0, max_population=30):
        self.num_epochs = num_epochs
        self.cull_ratio = cull_ratio
        self.max_population = max_population

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


    def test_epoch(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        inf_time = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # We use the profiler to measure inference time as it gives the most precise measurements
                # The overhead it introduces isn't too important as training accounts for the vast majority of the search algorithm
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

    def load_data(self):
        # Define transformation for training set
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Define transformation for testing set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Load CIFAR10 dataset
        trainset = CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_train)
        testset = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_test)

        # Create DataLoaders for training and testing sets
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        return trainloader, testloader

    def evaluate_model(self, model_cls):
        model = model_cls()

        # Fetch the metrics from the current population and calculate the
        # percentiles necessary at each epoch to terminate early
        open_mode = os.O_RDONLY | os.O_CREAT
        fd = os.open('pastepacc.txt', open_mode)
        pid = os.getpid()
        fcntl.flock(fd, fcntl.LOCK_SH)
        file_info = os.read(fd, os.path.getsize(fd)).decode().rstrip()
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        past_metrics = []
        for epoch in range(self.num_epochs-1):
            past_metrics.append([])
        if len(file_info) > 0:
            for line in file_info.split('\n'):
                metric, epoch = line.split()
                past_metrics[int(epoch)].append(float(metric))
        thresholds = [0]*(self.num_epochs-1)
        for epoch in range(self.num_epochs-1):
            past_metrics[epoch] = past_metrics[epoch][-self.max_population:]
            past_metrics[epoch].sort()
            if len(past_metrics[epoch]) >= max(self.max_population, self.num_epochs-epoch):
                target_rank = int(self.cull_ratio*self.max_population*(epoch+1)/(self.num_epochs-1))
                thresholds[epoch] = (past_metrics[epoch][target_rank]+past_metrics[epoch][target_rank-1])/2

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader, test_loader = self.load_data()

        # The metric ACC*inf_time**-0.07 is taken from the FBNet paper
        # This formula makes it so that an increase of 5% in accuracy and halving
        # the inference time have the same impact.

        for epoch in range(self.num_epochs):
            self.train_epoch(model, device, train_loader, optimizer, epoch)
            accuracy, inf_time = self.test_epoch(model, device, test_loader)
            metric = accuracy*(inf_time**-.07)
            nni.report_intermediate_result(accuracy*(inf_time**-.07))
            # Save the current metric into the file containing the metrics of the
            # whole population
            if epoch != self.num_epochs-1:
                open_mode = os.O_RDWR | os.O_CREAT
                fd = os.open('pastepacc.txt', open_mode)
                pid = os.getpid()
                fcntl.flock(fd, fcntl.LOCK_SH)
                file_info = os.read(fd, os.path.getsize(fd)).rstrip()
                os.write(fd, f"{metric} {epoch}\n".encode())
                # Save the metrics for future epochs as well if we terminate early
                if thresholds[epoch] > metric:
                    for future_epoch in range(epoch+1, self.num_epochs-1):
                        os.write(fd, f"{metric} {epoch}\n".encode())
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

                if thresholds[epoch] > metric:
                    break

        print(f'Final metric: {accuracy*(inf_time**-.07):.5f}')
        nni.report_final_result(accuracy*(inf_time**-.07))
