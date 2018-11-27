

# Impotring libraries we need
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt

# Mini-batch size(mini-batch gradient descent) . recommended values
# 64 - very large(65000 images)
# 16 - small (5000 images)
mb_size = 64

# This will transform data to tensor format which is pytorch's expexted format
transform = transforms.ToTensor()

# Here we download the dataset and transforms it, train=True will only download traning dataset 
# (train = True. Only need train data)
traindata = torchvision.datasets.MNIST('./NewData/', download=True, transform=transform, train = True)

# Loading the training data . Take train data , shuffle it , read/feed (batch-64 images) at a time
trainloader = torch.utils.data.DataLoader(traindata, shuffle=True, batch_size=mb_size)

# Just as an example we are going to visualize 
# We define an iterator 
dataiter = iter(trainloader) ## create an iterator object
imgs, labels = dataiter.next()

def imshow(imgs):
    """ Visualizing images """
    # make_grid will make a grid out of images
    imgs = torchvision.utils.make_grid(imgs)
    # transfoming tensors to numpy arrays to plot (matplot uses numpy)
    npimgs = imgs.numpy() ## torch.from_numpy() : other way
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimgs, (1,2,0)), cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
# imshow(imgs)
# print(imgs.size()) # 64(number of images) X 1 X 28(height) X 28(width)

# Defining the parematers of the network
h_dim = 128    # number of hidden neurons in our hidden layer
Z_dim = 100    # dimension of the input noise for generator (each image has noise)
lr = 1e-3      # learning rate
## flatten the image . view->reshape in numpy .
# imgs.size(0) -> number of images
X_dim = imgs.view(imgs.size(0), -1).size(1) 
# print(X_dim)# 28*28 =784

## Initialize weights(like apply for pandas)
def xavier_init(m):
    """ Xavier initialization """
    if type(m) == nn.Linear: ## Linear hidden layer
        nn.init.xavier_uniform_(m.weight) 
        m.bias.data.fill_(0) # fill it with zero , bias doesnt matter

# def xavier_uniform_(size):
#     in_dim = size[0] ## number of elements
#     # xavier variance is 1./in_dim
#     # For ReLu it is 2./in_dim
#     xavier_stddev = np.sqrt(2./in_dim)
#     return Variable(torch.randn(*size))*xavier_stddev

# Defining the Generator 
class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Z_dim, h_dim), ## Input -> noise,hidden layer(100 features)
            nn.ReLU(), ## Needed non linear activation 
            nn.Linear(h_dim, X_dim),
            nn.Sigmoid() # Binary cross entropy -> last layer , keep [0,1] 
        )
        self.model.apply(xavier_init)
    
    def forward(self, input):
        return self.model(input)
## 100 input features, feeded to 128 Hidden layers, with ReLu as activation func. 
## 128 hidden layers, output 784 
    
# Defining the Discriminator
class Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_dim, h_dim), ## input size always 784, fake and real
            nn.ReLU(),
            nn.Linear(h_dim, 1), # throw out number between 0 and 1
            nn.Sigmoid()
        )
        self.model.apply(xavier_init)
    
    def forward(self, input):
        return self.model(input)


# Instantiating the networks
G = Gen()
D = Dis()

# Defining solver to do the mini batch stochastic gradient descent 
# one for each network . Different solvers(one maximize, one minimize)
# Adam optimization algorithm
G_solver = opt.Adam(G.parameters(), lr = lr) 
D_solver = opt.Adam(D.parameters(), lr = lr)


# Defining the training for loop
for epoch in range(20):
    
    G_loss_run = 0.0
    D_loss_run = 0.0

    for i,data in enumerate(trainloader):
        X, _ = data #throw out your labels
        X = X.view(X.size(0), -1)
        mb_size = X.size(0) ## keep track of batch-size, if ,last batch less than 64
        
        # Defining labels for real (1s) and fake (0s) images
        one_labels = torch.ones(mb_size, 1)
        zero_labels = torch.zeros(mb_size, 1)
        
        # Random normal distribution for each image
        z = torch.randn(mb_size, Z_dim)
        
        # Feed forward in discriminator both 
        # fake and real images
        D_real = D(X) ##  real images -> Discriminator -> Outputs numbers[0,1]
        # fakes = G(z)
        D_fake = D(G(z)) # Feed noise to Generator to generate fake images
        
        # Defining the loss for Discriminator(teach network)
        # Gradient ascent (maximize loss)
        D_real_loss = F.binary_cross_entropy(D_real, one_labels) #push real images to 1
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels) #push fake images to 0
        D_loss = D_fake_loss + D_real_loss
        
        # backward propagation for discriminator
        D_solver.zero_grad() # recalculate the gradients (forget last time)
        D_loss.backward() # backward propogation
        D_solver.step() # take a step
        

        # Feed forward for generator
        z = torch.randn(mb_size, Z_dim) # New noise
        D_fake = D(G(z)) # New fake images
        
        # loss function of generator
        G_loss = F.binary_cross_entropy(D_fake, one_labels) #competition starts
        
        # backward propagation for generator
        G_solver.zero_grad() #zero the gradient
        G_loss.backward() #do backward propogation
        G_solver.step() #take a step and update bias and weights 
        
        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()
        
    # printing loss after each epoch 
    print('Epoch:{},   G_loss:{},   D_loss:{}'.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1) )) # average loss over data
    
    # Plotting fake images generated after each epoch by generator
    samples = G(z).detach() # detach from network , helps not mess with weights
    samples = samples.view(samples.size(0), 1, 28, 28)
    imshow(samples)