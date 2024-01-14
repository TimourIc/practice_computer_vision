import matplotlib.pyplot as plt 
import yaml
import pickle
from itertools import product

# IMPORT REPO PATHS
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

models_path = config["PATHS"]["models_path"]
figure_path=config["PATHS"]["figures_path"]
dataset_names=["MNIST", "CIFAR100"]
model_names = ["FNN", "AlexNet"]
training_loss_paths= [ f"{models_path}/{dataset_name}/{model_name}_model/{dataset_name}_training_loss.pickle" for (model_name, dataset_name) in product(model_names, dataset_names)]


def load_results(dataset_names, model_names):

    train_dict={ (model_name, dataset_name):{} for (model_name, dataset_name) in product(model_names, dataset_names) } 
    for dataset_name in dataset_names:
        for model_name in model_names:
            pickle_path=f"{models_path}/{dataset_name}/{model_name}_model/{dataset_name}_training_loss.pickle"
            with open(pickle_path, 'rb') as f:
                train_dict[(model_name, dataset_name)]= pickle.load(f)
    return train_dict

def plot_training_curves(train_loss, val_loss, test_loss):

    epochs=range(0,len(train_loss))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label= "validation loss")
    plt.legend()
    plt.plot()
    plt.show()
    

if __name__=="__main__":

    train_dict= load_results(dataset_names, model_names)

    model_name=model_names[0]
    for dataset_name in dataset_names:
        train_loss=train_dict[(model_name,dataset_name)]["training_loss"]
        val_loss=train_dict[(model_name,dataset_name)]["validation_loss"]
        test_loss=train_dict[(model_name,dataset_name)]["test_loss"]
        plot_training_curves(train_loss, val_loss, test_loss)

 