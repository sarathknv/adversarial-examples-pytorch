from torchvision import datasets, transforms


def load_dataset(dataset_name):

    if dataset_name == 'mnist':

        num_classes = 10
        in_channels = 1

        train = datasets.MNIST('../data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1307,), (0.3081,))
                                    #transforms.Normalize((0.5,), (0.5,))
                                    ]))


        test = datasets.MNIST('../data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1307,), (0.3081,))
                                    #transforms.Normalize((0.5,), (0.5,))
                                    ]))


    elif dataset_name == 'fmnist':

        num_classes = 10
        in_channels = 1

        train = datasets.FashionMNIST('../data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ]))

        test = datasets.FashionMNIST('../data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ]))


    elif dataset_name == 'cifar10':

        num_classes = 10
        in_channels = 3

        train = datasets.CIFAR10('../data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
                                    ]))

        test = datasets.CIFAR10('../data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ]))

    return (train, test, in_channels, num_classes)
