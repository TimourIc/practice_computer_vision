import matplotlib.pyplot as plt
import yaml

from experiments.plotting.plot_functions import FigureMaker

# IMPORT REPO PATHS
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

models_path = config["PATHS"]["models_path"]
figure_path = config["PATHS"]["figures_path"]
dataset_names = ["MNIST", "CIFAR100"]
model_names = ["FNN", "AlexNet"]


if __name__ == "__main__":
    figmaker = FigureMaker(dataset_names=dataset_names, model_names=model_names)
    figmaker.load_results(models_path=models_path)
    figmaker.plot_training_curves(
        save_path=figure_path, figure_name="training_features"
    )
    figmaker.plot_test_results(save_path=figure_path, figure_name="test_comparison")
