.DEFAULT_GOAL := help
TAG ?= rr5555/neuroai-project:jammy-cuda


docker-build-cuda: ## Build cuda Docker img
	docker build --tag $(TAG) ./Docker/cuda

docker-push-cuda: ## Push cuda Docker img
	docker push $(TAG)

docker-run-cuda: ## Run cuda Docker container
	docker run --gpus all -v .:/root/neuroAI-project -itd --name neuroAI-project $(TAG)

# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
