import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np


class FigureMaker:
    def __init__(self, dataset_names: list, model_names: list):
        self.dataset_names = dataset_names
        self.model_names = model_names

    def load_results(self, models_path: str) -> dict:
        train_dict = {
            (model_name, dataset_name): {}
            for (model_name, dataset_name) in product(
                self.model_names, self.dataset_names
            )
        }
        for dataset_name in self.dataset_names:
            for model_name in self.model_names:
                pickle_path = f"{models_path}/{dataset_name}/{model_name}_model/training_loss.pickle"
                with open(pickle_path, "rb") as f:
                    train_dict[(model_name, dataset_name)] = pickle.load(f)
        self.train_dict = train_dict

    def plot_training_curves(self, save_path: str, figure_name: str):
        fig, axs = plt.subplots(2, len(self.dataset_names), figsize=(8, 6))
        colors = ["blue", "red", "green"]

        for i, dataset_name in enumerate(self.dataset_names):
            for j, model_name in enumerate(self.model_names):
                train_loss = self.train_dict[(model_name, dataset_name)][
                    "training_loss"
                ]
                val_loss = self.train_dict[(model_name, dataset_name)][
                    "validation_loss"
                ]
                train_accuracy = self.train_dict[(model_name, dataset_name)][
                    "training_accuracy"
                ]
                val_accuracy = self.train_dict[(model_name, dataset_name)][
                    "validation_accuracy"
                ]
                test_loss = self.train_dict[(model_name, dataset_name)]["test_loss"]

                epochs = range(0, len(train_loss))
                axs[0, i].plot(
                    epochs,
                    train_loss,
                    label=f"{model_name}: train loss",
                    color=colors[j],
                )
                axs[0, i].plot(
                    epochs,
                    val_loss,
                    label=f"{model_name}: val loss",
                    linestyle="--",
                    color=colors[j],
                )
                axs[1, i].plot(
                    epochs,
                    train_accuracy,
                    label=f"{model_name}: train acc.",
                    color=colors[j],
                )
                axs[1, i].plot(
                    epochs,
                    val_accuracy,
                    label=f"{model_name}: val acc.",
                    linestyle="--",
                    color=colors[j],
                )

            if i==0:
                axs[0, i].legend(loc="upper right", fontsize=8)
            # axs[1, i].legend(loc="upper right", fontsize=7)
            axs[0, i].set_title(f"{dataset_name}")
            axs[1, i].set_title(f"{dataset_name}")
            axs[1, i].set_xlabel("epochs")
            axs[0, 0].set_ylabel("loss")
            axs[1, 0].set_ylabel("accuracy")

        plt.tight_layout()
        fig.savefig(f"{save_path}/{figure_name}")

    def plot_test_results(self, save_path: str, figure_name: str):
        bar_width = 0.35
        index = np.arange(len(self.dataset_names))
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for i, model_name in enumerate(self.model_names):
            bars = ax.bar(
                index + i * bar_width,
                [
                    self.train_dict[(model_name, dataset_name)]["test_accuracy"]
                    for dataset_name in self.dataset_names
                ],
                bar_width,
                label="a",
            )

            # Add text labels on top of the bars
            for bar in (bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{model_name} {100*height:.1f}%",
                    ha="center",
                    va="bottom",
                )

        ax.set_xticks(index + bar_width * (len(self.dataset_names) - 1) )
        self.dataset_names=["fashion MNIST" if i=="MNIST" else i for i in self.dataset_names   ]
        ax.set_xticklabels(self.dataset_names)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        fig.savefig(f"{save_path}/{figure_name}")
