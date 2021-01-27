To train the model, run with the following command line arguments, making the correct substitutions:

train <training file name> <output model name>

Where training file name is the csv file you wish to use for training, and output model name is any name you choose for the model.



To predict, run with the following command line arguments, making the correct substitutions:

predict <model name> <file name to predict>

where the model name is the saved model from training, and the file name to predict is a test file to calculate predictions.




Note!!
There are already 2 trained models in this directory, named model_from_total and model_from_subsample. 
The first is a model trained using Frogs.csv, and the other is trained using Frogs-subsample.csv