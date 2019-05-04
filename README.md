# CS 398 Deep Learning @ UIUC

## Homework 4 Deep Convolution Neural Network for CIFAR10 Dataset

Name: Yawen Duan		UIN: 655877290

### **HW4 Description:**

Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network should use **(A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation**. For 10% extra credit, compare dropout test accuracy **(i) using the heuristic prediction rule and (ii) Monte Carlo simulation**. For full credit, the model should achieve 80-90% Test Accuracy. Submit via Compass (1) the code and (2) a paragraph (in a PDF document) which reports the results and briefly describes the model architecture. 

### Implementation

In my code, I defined an object  `Net` to obtain the model to be used. The hyperparameters are listed as follows:

```python
batch_size = 50
learning_rate = 0.001
num_workers = 4
# Note that we can obtain the pretrained model to reduce training time
load_pretrained_model = True
epochs = 50
pretrained_epoch = 15
```

The model consists of 5 convolutional layers and 3 fully-connected layers, with max-pooling and dropouts between some layers. The structure of the model can be shown as follows:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=144, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=144, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.Dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*256
        x = F.relu(self.conv2(x)) #32*32*512
        x = self.pool(x) #16*16*512
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*256
        x = F.relu(self.conv4(x)) #16*16*128
        x = self.pool(x) # 8*8*128
        x = self.Dropout(x)
        x = F.relu(self.conv5(x)) #8*8*128
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x
```

In this model, I apply data augmentation to the dataset through random horizontal flip. This model is trained with ADAM optimizer.

### Test Result 

The test accuracy and value of loss function with respect to the number of epochs are shown as follows. Note the test has achieved an accuracy score of above 83% after 15 epochs.

![Alt text](assets/test_output.png/?raw=true "Test Accuracy")
