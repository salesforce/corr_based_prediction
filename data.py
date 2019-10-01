
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import _split_train_val
import torchvision.datasets as datasets
import torch.utils.data as utils
import errno
from PIL import Image

torch.manual_seed(0)

NUM_WORKERS = 0

def get_dataset(args):
	if args.dataset=='mnist':
		trans = ([ transforms.ToTensor()]) 
		trans = transforms.Compose(trans)
		fulltrainset = torchvision.datasets.MNIST(root=args.data, train=True, transform=trans, download=True)

		train_set, valset = _split_train_val(fulltrainset, val_fraction=0.1)


		trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True,
		                                          num_workers=NUM_WORKERS, pin_memory=True)
		validloader = torch.utils.data.DataLoader(valset, batch_size=args.bs, shuffle=False,
		                                          num_workers=NUM_WORKERS, pin_memory=True)


		test_set = torchvision.datasets.MNIST(root=args.data, train=False, transform=trans)
		testloader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=NUM_WORKERS)

		nb_classes = 10
		dim_inp=28*28 # np.prod(train_set.data.size()[1:])
	elif 'cmnist' in args.dataset:
		data_dir_cmnist = args.data + 'cmnist/' + args.dataset + '/'
		data_x = np.load(data_dir_cmnist+'train_x.npy')
		data_y = np.load(data_dir_cmnist+'train_y.npy')

		data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
		data_y = torch.from_numpy(data_y).type('torch.LongTensor')

		my_dataset = utils.TensorDataset(data_x,data_y)

		train_set, valset = _split_train_val(my_dataset, val_fraction=0.1)

		trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=NUM_WORKERS)
		validloader = torch.utils.data.DataLoader(valset, batch_size=args.bs, shuffle=False,
		                                          num_workers=NUM_WORKERS, pin_memory=True)


		data_x = np.load(data_dir_cmnist+'test_x.npy')
		data_y = np.load(data_dir_cmnist+'test_y.npy')
		data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
		data_y = torch.from_numpy(data_y).type('torch.LongTensor')
		my_dataset = utils.TensorDataset(data_x,data_y)
		testloader = torch.utils.data.DataLoader(my_dataset, batch_size=args.bs, shuffle=False, num_workers=NUM_WORKERS)

		nb_classes = 10
		dim_inp=28*28* 3
	elif args.dataset=='mnistm':
		trans = ([transforms.ToTensor()]) 
		trans = transforms.Compose(trans)
		fulltrainset = MNISTM(root=args.data, train=True, transform=trans, download=True)

		train_set, valset = _split_train_val(fulltrainset, val_fraction=0.1)


		trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True,
		                                          num_workers=2, pin_memory=True)
		validloader = torch.utils.data.DataLoader(valset, batch_size=args.bs, shuffle=False,
		                                          num_workers=2, pin_memory=True)


		test_set = MNISTM(root=args.data, train=False, transform=trans)
		testloader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=2)

		nb_classes = 10
		dim_inp=3*28*28 # np.prod(train_set.data.size()[1:])
	elif args.dataset=='svhn':
		trans = ([torchvision.transforms.Resize((28,28), interpolation=2), transforms.ToTensor()]) 
		trans = transforms.Compose(trans)
		fulltrainset = torchvision.datasets.SVHN(args.data, split='train', transform=trans, target_transform=None, download=True)

		train_set, valset = _split_train_val(fulltrainset, val_fraction=0.1)


		trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True,
		                                          num_workers=NUM_WORKERS, pin_memory=True)
		validloader = torch.utils.data.DataLoader(valset, batch_size=args.bs, shuffle=False,
		                                          num_workers=NUM_WORKERS, pin_memory=True)


		test_set = torchvision.datasets.SVHN(args.data, split='test', transform=trans, target_transform=None, download=True)
		testloader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=NUM_WORKERS)

		nb_classes = 10
		dim_inp=3*28*28 
	return trainloader, validloader, testloader, nb_classes, dim_inp


class MNISTM(torch.utils.data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self,
                 root, mnist_root="data",
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.training_file))
        else:
            self.test_data, self.test_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
            os.path.exists(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print('Downloading ' + self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace('.gz', '')):
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), "rb") as f:
            mnist_m_data = pickle.load(f, encoding='bytes')
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root,
                                            train=True,
                                            download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root,
                                           train=False,
                                           download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')
