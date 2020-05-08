import mlpredict


VGG16 = mlpredict.api.dnn(input_dimension=3, input_size=224)


VGG16.add_layer('Convolution', 'conv1_1', kernelsize=3, channels_out=64, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'conv1_2', kernelsize=3, channels_out=64, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Max_pool', 'pool1', pool_size=2, padding='SAME', strides=2)


VGG16.add_layer('Convolution', 'conv2_1', kernelsize=3, channels_out=128, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'conv2_2', kernelsize=3, channels_out=128, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Max_pool', 'pool2', pool_size=2, padding='SAME', strides=2)


VGG16.add_layer('Convolution', 'conv3_1', kernelsize=3, channels_out=256, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'conv3_2', kernelsize=3, channels_out=256, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'conv3_3', kernelsize=3, channels_out=256, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Max_pool', 'pool3', pool_size=2, padding='SAME', strides=2)


VGG16.add_layer('Convolution', 'conv4_1', kernelsize=3, channels_out=512, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'conv4_2', kernelsize=3, channels_out=512, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'conv4_3', kernelsize=3, channels_out=512, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Max_pool', 'pool4', pool_size=2, padding='SAME', strides=2)


VGG16.add_layer('Convolution', 'conv5_1', kernelsize=3, channels_out=512, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'conv5_2', kernelsize=3, channels_out=512, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'conv5_3', kernelsize=3, channels_out=512, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Max_pool', 'pool5', pool_size=2, padding='SAME', strides=2)


VGG16.add_layer('Convolution', 'fc6', kernelsize=7, channels_out=4096, 
                padding='VALID', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'fc7', kernelsize=1, channels_out=4096, 
                padding='SAME', strides=1, use_bias=1, activation='relu')
VGG16.add_layer('Convolution', 'fc8', kernelsize=1, channels_out=1000, 
                padding='SAME', strides=1, use_bias=1, activation='relu')

VGG16.describe()
VGG16.save('./VGG16.json')
