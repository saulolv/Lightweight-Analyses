.PHONY: train train-cifar10 train-cifar100 train-wakevision
.PHONY: corruption corruption-cifar10 corruption-cifar100 corruption-wakevision
.PHONY: robustness figures latency stats clean help

NOTEBOOKS := notebooks

help:
	@echo "Available targets:"
	@echo "  train-cifar10        Train & export CIFAR-10 models"
	@echo "  train-cifar100       Train & export CIFAR-100 models"
	@echo "  train-wakevision    Train & export Wake Vision models"
	@echo "  corruption-cifar10   Run CIFAR-10-C corruption evaluation"
	@echo "  corruption-cifar100  Run CIFAR-100-C corruption evaluation"
	@echo "  corruption-wakevision Run Wake Vision corruption evaluation"
	@echo "  robustness           Aggregate mCE / relative mCE"
	@echo "  stats               Friedman + Nemenyi statistical tests"
	@echo "  latency             Generate latency/throughput figures"
	@echo "  figures             Run all figure-generating notebooks"
	@echo "  docker-notebooks    Build notebook Docker image"
	@echo "  docker-pi           Build Raspberry Pi Docker image"
	@echo "  clean               Remove generated artifacts"

train-cifar10:
	jupyter execute $(NOTEBOOKS)/01_cifar10_training_export.ipynb

train-cifar100:
	jupyter execute $(NOTEBOOKS)/02_cifar100_training_export.ipynb

train-wakevision:
	jupyter execute $(NOTEBOOKS)/03_wake_vision_training.ipynb
	jupyter execute $(NOTEBOOKS)/04_wake_vision_mobile_export.ipynb

corruption-cifar10:
	jupyter execute $(NOTEBOOKS)/05_cifar10c_corruption_error.ipynb

corruption-cifar100:
	jupyter execute $(NOTEBOOKS)/06_cifar100c_corruption_error.ipynb

corruption-wakevision:
	jupyter execute $(NOTEBOOKS)/07_wake_vision_robustness.ipynb

robustness:
	jupyter execute $(NOTEBOOKS)/08_robustness_results.ipynb

stats:
	jupyter execute $(NOTEBOOKS)/09_statistical_tests.ipynb

latency:
	jupyter execute $(NOTEBOOKS)/10_latency_analyses.ipynb

figures: robustness stats latency
	@echo "All figures generated in figures/"

docker-notebooks:
	docker build -f Dockerfile.notebooks -t lightweight-cnn-notebooks .

docker-pi:
	docker build -t lightweight-pi Raspberry/

clean:
	rm -rf figures/*.pdf notebooks/output/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true