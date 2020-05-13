本作业的目标如下：

理解神经网络及其分层结构。
理解并实现（向量化）反向传播。
实现多个用于神经网络最优化的更新方法。
实现用于训练深度网络的批量归一化（ batch normalization ）。
实现随机失活（dropout）。
进行高效的交叉验证并为神经网络结构找到最好的超参数。
理解卷积神经网络的结构，并积累在数据集上训练此类模型的经验。

Q1：全连接神经网络（30分）FullyConnectedNets.ipynb
Q2：批量归一化（30分）BatchNormalization.ipynb
Q3：随机失活（Dropout）（10分）Dropout.ipynb
Q4：在CIFAR-10上运行卷积神经网络（30分）ConvolutionalNetworks.ipynb



**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the `assignment2` directory:

```bash
cd cs231n/datasets
./get_datasets.sh
```

**Compile the Cython extension:** 
Convolutional Neural Networks require a very
efficient implementation. We have implemented of the functionality using
[Cython](http://cython.org/); you will need to compile the Cython extension
before you can run the code. From the `cs231n` directory, run the following
command:

```bash
python setup.py build_ext --inplace
```


### Q1: Fully-connected Neural Network (30 points)
The IPython notebook `FullyConnectedNets.ipynb` will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

### Q2: Batch Normalization (30 points)
In the IPython notebook `BatchNormalization.ipynb` you will implement batch
normalization, and use it to train deep fully-connected networks.

### Q3: Dropout (10 points)
The IPython notebook `Dropout.ipynb` will help you implement Dropout and explore
its effects on model generalization.

### Q4: ConvNet on CIFAR-10 (30 points)
In the IPython Notebook `ConvolutionalNetworks.ipynb` you will implement several
new layers that are commonly used in convolutional networks. You will train a
(shallow) convolutional network on CIFAR-10, and it will then be up to you to
train the best network that you can.

### Q5: Do something extra! (up to +10 points)
In the process of training your network, you should feel free to implement
anything that you want to get better performance. You can modify the solver,
implement additional layers, use different types of regularization, use an
ensemble of models, or anything else that comes to mind. If you implement these
or other ideas not covered in the assignment then you will be awarded some bonus
points.

