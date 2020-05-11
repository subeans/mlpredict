import mlpredict


MLP = mlpredict.api.dnn(input_dimension=1, input_size=28)


MLP.add_layer('Fully_connected','hidden_layer',activation='relu',neurons='128')
MLP.add_layer('Fully_connected', 'output_layer', activation='softmax',neurons='10')


MLP.describe()
MLP.save('./MLP.json')
