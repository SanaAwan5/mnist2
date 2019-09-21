# import libraries
import torch
import numpy as np
#import matplotlib.pyplot as plt
import time

from torchvision import datasets
import torchvision.transforms as transforms
#import torch.nn as nn
#import torch.nn.functional as F
import warnings
import numpy as np
import phe as paillier
import time
from sonar.contracts_listclass_unencrypted import ModelRepository,Model,Gradient_List
from syft.he.paillier.keys import KeyPair,SecretKey,PublicKey
from syft.nn.linear import LinearClassifier
from sklearn.datasets import load_breast_cancer
from mlxtend.data import loadlocal_mnist
from itertools import cycle
def get_balance(account):
    return repo.web3.fromWei(repo.web3.eth.getBalance(account),'ether')

warnings.filterwarnings('ignore')

"""X, y = loadlocal_mnist(
        images_path='/Users/Sana/Desktop/mnist/PySonar/train-images-idx3-ubyte', 
        labels_path='/Users/Sana/Desktop/mnist/PySonar/train-labels-idx1-ubyte')
print('Dimensions: %s x %s' % (X.shape[0],X.shape[1]))
print('\n1st row', X[0])"""



#from torch.utils.data import sampler
from sklearn.datasets import load_diabetes
diabetes = load_breast_cancer()
y = diabetes.target
X = diabetes.data

#print (type(diabetes.data))



list1=[]
validation=(X[0:59],y[0:59])
anonymous_diabetes_user0=(X[101:150],y[101:150])
list1.append(anonymous_diabetes_user0)
print("len",len(list1[0][0]))
anonymous_diabetes_user1=(X[151:200],y[151:200])
list1.append(anonymous_diabetes_user1)
anonymous_diabetes_user2=(X[201:250],y[201:250])
list1.append(anonymous_diabetes_user2)
anonymous_diabetes_user3=(X[251:300],y[251:300])
list1.append(anonymous_diabetes_user3)
anonymous_diabetes_user4=(X[351:400],y[351:400])
list1.append(anonymous_diabetes_user4)
anonymous_diabetes_user5=(X[451:500],y[451:500])
list1.append(anonymous_diabetes_user5)
anonymous_diabetes_user6=(X[501:550],y[501:550])
list1.append(anonymous_diabetes_user6)
anonymous_diabetes_user7=(X[101:150],y[101:150])
list1.append(anonymous_diabetes_user7)
anonymous_diabetes_user8=(X[151:200],y[151:200])
list1.append(anonymous_diabetes_user8)
anonymous_diabetes_user9=(X[201:250],y[201:250])
list1.append(anonymous_diabetes_user9)


#list1.append(anonymous_diabetes_user0,anonymous_diabetes_user1,anonymous_diabetes_user2,anonymous_diabetes_user3,anonymous_diabetes_user4,anonymous_diabetes_user5,anonymous_diabetes_user6,anonymous_diabetes_user7,anonymous_diabetes_user8,anonymous_diabetes_user9)

#print("anon shape", anonymous_diabetes_users[0].shape)



# prepare data loaders
start=time.time()

# we're also going to initialize the model trainer smart contract, which in the
# real world would already be on the blockchain (managing other contracts) before
# the simulation begins

# ATTENTION: copy paste the correct address (NOT THE DEFAULT SEEN HERE) from truffle migrate output.
repo = ModelRepository('0xBE30EC73A0b86b2632783EA5414cE07df7C94be6') # blockchain hosted model repository

# we're going to set aside 10 accounts for our 42 patients
# Let's go ahead and pair each data point with each patient's 
# address so that we know we don't get them confused
patient_addresses = repo.web3.eth.accounts[2:11]
#anonymous_diabetics = list(zip(cycle(patient_addresses),anonymous_diabetes_users[0],anonymous_diabetes_users[1]))
#print("validation[0].size",len(anonymous_diabetics))
#anonymous_diabetics=list(zip(cycle(patient_addresses),anonymous_diabetes_users[0],anonymous_diabetes_users[1]))

# we're going to set aside 1 account for Cure Diabetes Inc
cure_diabetes_inc = repo.web3.eth.accounts[0]
agg_addr = repo.web3.eth.accounts[1]

pubkey,prikey = KeyPair().generate(n_length=1024)
#pubkey,prikey=paillier.paillier.generate_paillier_keypair()
diabetes_classifier = LinearClassifier(desc="DiabetesClassifier",n_inputs=30,n_labels=1)
#initial_error = diabetes_classifier.evaluate(validation[0],validation[1])
#diabetes_classifier.encrypt(pubkey)
s1,s2=paillier.paillier.genKeyShares(prikey.sk,pubkey.pk)
st=SecretKey(s1)
sab=SecretKey(s2)
s3,s4=paillier.paillier.genKeyShares(s2,pubkey.pk)
sa=SecretKey(s3)
scb=SecretKey(s4)

diabetes_model = Model(owner=cure_diabetes_inc,
                       syft_obj = diabetes_classifier,
                       bounty = 10,
                       initial_error = 210,
                       target_error =0 ,
                       best_error= 210
                      )
model_id = repo.submit_model(diabetes_model)
#print('initial error',initial_error)
model=repo[model_id]
local_losses=0
alpha = 0.09
j=0


#new_error = repo[model_id].syft_obj.evaluate(images,labels)
new_error = model.initial_error
print ("new_error", model.initial_error)

  


   # train_epochs = 10
batch_size = 50
training_round =0

    #for epoch in range (train_epochs):

while new_error  > model.target_error:
    for k in range(len(list1)):
        if (new_error <= model.target_error):
                    print("broken round %s %s", i/64, training_round)
                    model.update(addr, new_error,candidate,pubkey)
                    break
        addr = repo.web3.eth.accounts[k]
        len_n=[]
        for i,(data, target) in enumerate(list1):
            if (i == 0):
                print("batch_index", i)
                old_balance = get_balance(addr)
                print('model_id',model.model_id)
                gradient,candidate,len_n=model.generate_gradient(addr,prikey,data,target,alpha,i)
                print("number of gradients of model",len(model))
        #local_losses=candidate.evaluate(validation[0],validation[1])
        #local_loss=model.evaluate_gradient(addr, gradient, prikey, pubkey,validation[0],validation[1], alpha)
        #print("local loss",local_losses)
                model.submit_transformed_gradients(gradient,pubkey,st)   
            if (i%50 ==0):
                training_round = training_round + 1
                gradient_list=Gradient_List(model.model_id, repo=repo, model=model)
        #gradient_list=gradient_list[model_id]
                avg_gradient=gradient_list.generate_gradient_avg(addr,sa,alpha)
                decrypted_avg=model.decrypt_avg(scb)
                new_error,candidate = model.evaluate_gradient_from_avg(agg_addr,decrypted_avg,prikey,pubkey,validation[0],validation[1],alpha)
    #repo[model_id].syft_obj.decrypt(prikey)
                if (new_error <= model.target_error):
                    print("broken round %s %s", i/64, training_round)
                    model.update(addr, new_error,candidate,pubkey)
                    break
                model.update(addr,new_error,candidate,pubkey)
                print("model best error", model.best_error)
                updatedModel=model
                print("updated model's best error", updatedModel.best_error)
                print("new error from averaged gradients = "+str(new_error))
                incentive = (get_balance(addr) - old_balance)
                print("incentive = "+str(incentive))
                break
            else:
                print("batch_index", i)
                old_balance = get_balance(addr)
                print('model_id',model.model_id)
                gradient,candidate=model.generate_gradient(addr,prikey,data,target,alpha)
                print("number of gradients of model",len(model))
            #local_losses=candidate.evaluate(validation[0],validation[1])
            #local_loss=model.evaluate_gradient(addr, gradient, prikey, pubkey,validation[0],validation[1], alpha)
            #print("local loss",local_losses)
                model.submit_transformed_gradients(gradient,pubkey,st)
"""if (i % 5==0):
            print("list data len",data)
            old_balance = get_balance(addr)
            print('model_id',model.model_id)
            gradient,candidate,len=model.generate_gradient(addr,prikey,data,target,alpha)
            len_n.append(lenn)
            print("number of gradients of model",len(model))
        #local_losses=candidate.evaluate(validation[0],validation[1])
        #local_loss=model.evaluate_gradient(addr, gradient, prikey, pubkey,validation[0],validation[1], alpha)
        #print("local loss",local_losses)
            model.submit_transformed_gradients(gradient,pubkey,st)   
        gradient_list=Gradient_List(model.model_id, repo=repo, model=model)
    #gradient_list=gradient_list[model_id]
        avg_gradient=gradient_list.generate_gradient_avg(addr,sa,len_n,alpha)
        decrypted_avg=model.decrypt_avg(scb)
        new_error,candidate = model.evaluate_gradient_from_avg(agg_addr,decrypted_avg,prikey,pubkey,validation[0],validation[1],alpha)
    #repo[model_id].syft_obj.decrypt(prikey)
        model.update(addr,new_error,candidate,pubkey)
        print("model best error", model.best_error)
        updatedModel=model
        print("updated model's best error", updatedModel.best_error)
        print("new error from averaged gradients = "+str(new_error))
        incentive = (get_balance(addr) - old_balance)
        print("incentive = "+str(incentive))
        if (new_error <= model.target_error):
            print("broken round %s %s", i/64, training_round)
            model.update(addr, new_error,candidate,pubkey)
            break"""
        

        
        
 
end = time.time()
print('execution time', end - start)
    #if (new_error <= model.target_error):
      #  print("broken round", j)
       # break







"""#matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(str(labels[idx].item()))
img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')



# define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

# initialize the NN
model = Net()
print(model)
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model
n_epochs = 50

#model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for (data, target) in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
             
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))
"""