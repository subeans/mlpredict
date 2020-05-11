import mlpredict


MLP = mlpredict.api.dnn(input_dimension=1, input_size=28)


MLP.add_layer('Convolution', 'fc1', kernelsize=1, channels_out=128, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
MLP.add_layer('Convolution', 'fc1', kernelsize=1, channels_out=10 
                padding='SAME', strides=1, use_bias=1, activation='softmax')


MLP.describe()
MLP.save('./MLP.json')
