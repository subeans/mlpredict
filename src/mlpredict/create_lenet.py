import mlpredict


Lenet5 = mlpredict.api.dnn(input_dimension=1, input_size=28)


Lenet5.add_layer('Convolution', 'conv1_1', kernelsize=3, channels_out=6, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
                
                
Lenet5.add_layer('Max_pool', 'pool1', pool_size=2, padding='SAME', strides=2)


Lenet5.add_layer('Convolution', 'conv2_1', kernelsize=3, channels_out=16, 
                padding='SAME', strides=1, use_bias=1, activation='relu')

Lenet5.add_layer('Convolution', 'fc1', kernelsize=1, channels_out=120, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
Lenet5.add_layer('Convolution', 'fc2', kernelsize=1, channels_out=84, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
Lenet5.add_layer('Convolution', 'fc3', kernelsize=1, channels_out=10, 
                padding='SAME', strides=1, use_bias=1, activation='softmax')

Lenet5.describe()
Lenet5.save('mlpredict/dnn_architecture/Lenet5.json')
