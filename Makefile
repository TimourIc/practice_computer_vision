clean:
	black src
	black experiments
	isort src
	isort experiments
venv:
	python3 -m venv venv
clean_logs:
	rm -rf logs/*

FNN_tuning_MNIST:
	python3 -m experiments.tuning.main --MODEL_NAME FNN --DATASET_NAME MNIST 
AlexNet_tuning_MNIST:
	python3 -m experiments.tuning.main --MODEL_NAME AlexNet --DATASET_NAME MNIST
all_tuning_MNIST: FNN_tuning_MNIST AlexNet_tuning_MNIST

FNN_training_MNIST:
	python3 -m experiments.training.main --MODEL_NAME FNN --DATASET_NAME MNIST --READ_HP
AlexNet_training_MNIST:
	python3 -m experiments.training.main --MODEL_NAME AlexNet --DATASET_NAME MNIST --READ_HP 
all_training_MNIST: FNN_training_MNIST AlexNet_training_MNIST

tune_and_train_all_MNIST: all_tuning_MNIST all_training_MNIST


FNN_tuning_CIFAR100:
	python3 -m experiments.tuning.main --MODEL_NAME FNN --DATASET_NAME CIFAR100 
AlexNet_tuning_CIFAR100:
	python3 -m experiments.tuning.main --MODEL_NAME AlexNet --DATASET_NAME CIFAR100
all_tuning_CIFAR100: FNN_tuning_CIFAR100 AlexNet_tuning_CIFAR100

FNN_training_CIFAR100:
	python3 -m experiments.training.main --MODEL_NAME FNN --DATASET_NAME CIFAR100 --READ_HP
AlexNet_training_CIFAR100:
	python3 -m experiments.training.main --MODEL_NAME AlexNet --DATASET_NAME CIFAR100 --READ_HP
all_training_CIFAR100: FNN_training_CIFAR100 AlexNet_training_CIFAR100

tune_and_train_all_CIFAR100: all_tuning_CIFAR100 all_training_CIFAR100

make_figures:
	python3 -m experiments.plotting.main