import torch

from torch.utils.data import Dataset
import torchvision

import random
import numpy as np



class train_dataset(Dataset):

    def __init__(self , transform = None):
        # we use MNIST as our example
        mnist = torchvision.datasets.MNIST(root = "./data" , 
                                                    train = True,
                                                    transform=None,
                                                    download=True)

        train_x = mnist.data.numpy() / 255.0     # 60000,28,28
        train_y = mnist.targets.numpy()           # 60000,

        self.mean , self.std = train_x.mean() , train_x.std()
    


        self.pairs , self.labels = pairs_process(train_x , train_y)
        # print(self.pairs.shape)   # 108400,28,28
        # print(self.labels.shape)  # 108400,1

    
        self.n_samples = self.pairs.shape[0]    

        self.transform = transform


    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):

        if self.transform:
            inputs = self.transform(self.pairs[index] , self.mean , self.std)
        else:
            inputs = self.pairs[index]

        targets = self.labels[index]

        # print(torch.from_numpy(inputs).dtype)
        # print(torch.from_numpy(inputs).float().dtype)

        return torch.from_numpy(inputs).float() , torch.from_numpy(targets).float()



    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

    def get_statistics(self):
        return self.mean , self.std



# simply normalize
class normalize():

    def __call__(self, inputs , mean , std):

        return (inputs - mean) / (std + 1e-8)




class test_dataset(Dataset):

    def __init__(self , transform = None , statistics = None):
        # we use MNIST as our example
        mnist = torchvision.datasets.MNIST(root = "./data" , 
                                           train = False,
                                           transform=None,)

        test_x = mnist.data.numpy() / 255.0      # 60000,28,28
        test_y = mnist.targets.numpy()           # 60000,

        self.mean , self.std = statistics


        self.pairs , self.labels = pairs_process(test_x , test_y)
    
        self.n_samples = self.pairs.shape[0]    

        self.transform = transform


    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):

        if self.transform:
            inputs = self.transform(self.pairs[index] , self.mean , self.std)
        else:
            inputs = self.pairs[index]

        targets = self.labels[index]

        # print(torch.from_numpy(inputs).dtype)
        # print(torch.from_numpy(inputs).float().dtype)

        return torch.from_numpy(inputs).float() , torch.from_numpy(targets).float()



    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



def pairs_process(data_x , data_y):

    num_classes = 10

    digit_indices = [np.where(data_y == i)[0] for i in range(num_classes)]
    
    #create pairs
    pairs ,labels = [] , []

    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1

    for d in range(num_classes):
        for i in range(n):

            idx1 = digit_indices[d][i]
            idx2 = digit_indices[d][random.randint(0 , n-1)]

            pairs.append([data_x[idx1] , data_x[idx2]])
            labels.append([1.])  # matched successfully


            other_class =  (d + random.randint(1 , num_classes-1)) % num_classes  # randomly choose a class which is different from d
            idx3 = digit_indices[other_class][random.randint(0 , n-1)]
            
            pairs.append([data_x[idx1] , data_x[idx3]])
            labels.append([0.]) # matched failed
    

    return np.expand_dims(np.array(pairs) , axis = 2) , np.array(labels)
            # (n , 2 , 1 , H , W)
            # dim 0 = #samples, dim 1 = training pair for similarity matching, dim 2 = #input_channels



if __name__ == "__main__":


    a = torchvision.datasets.MNIST(root = "./data" , 
                                    train = True,
                                    transform=None,
                                    download=True)

    b = torchvision.datasets.MNIST(root = "./data" , 
                                    train = False,
                                    transform=None)


    ax = a.data.numpy()
    ay = a.targets.numpy()

    print(type(ax) , type(ay))     
    print(ax.shape)
    print(ay.shape)


    # print(ay[5])   # not yet one-hot coding, so it is just a single number
    # print(ax[5])   # not yet normalizer, so the range is from 0 to 255


    # D = train_dataset()
    # print(len(D))