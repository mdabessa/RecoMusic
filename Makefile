# RecoMusic — root automation.
# Pipeline targets need Docker + Docker Compose; deploy targets need terraform + awscli.

SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE      := docker compose
TF_DIR       := infra/terraform
FRONTEND_DIR := frontend
DIST_DIR     := $(FRONTEND_DIR)/dist
AWS_REGION   ?= us-east-1

.PHONY: help build smoke pipeline logs stop up down deploy invalidate cost clean \
        _require-tf _require-deploy

help: ## List available targets
	@echo "RecoMusic targets:"
	@grep -hE '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  %-11s %s\n", $$1, $$2}'

# --- phase 1: local compute -------------------------------------------------

build: ## Build the pipeline Docker image
	$(COMPOSE) build

smoke: ## Verify the pipeline image can import PySpark and start a local Spark session
	$(COMPOSE) run --rm --no-deps spark python -c "import pyspark; print('pyspark', pyspark.__version__)"
	$(COMPOSE) run --rm --no-deps spark python -c "from pyspark.sql import SparkSession; s = SparkSession.builder.master('local[2]').getOrCreate(); print('spark', s.version); s.stop()"

pipeline: build ## Run the full pipeline (ingest -> models -> artifacts) via Prefect
	@test -f pipeline/flows.py || { echo "pipeline/flows.py not found — Phase 1 not implemented yet"; exit 1; }
	$(COMPOSE) up -d prefect
	$(COMPOSE) run --rm spark python flows.py

logs: ## Tail logs of the local pipeline services
	$(COMPOSE) logs -f --tail=100

stop: ## Stop local containers (keeps volumes and data/)
	$(COMPOSE) down

# --- phase 5: AWS deploy ----------------------------------------------------

up: _require-deploy ## terraform apply, then build + upload the site
	terraform -chdir=$(TF_DIR) init -input=false
	terraform -chdir=$(TF_DIR) apply -auto-approve -input=false
	$(MAKE) deploy

deploy: _require-deploy ## Build the frontend, sync it to S3, invalidate CloudFront
	npm --prefix $(FRONTEND_DIR) ci
	npm --prefix $(FRONTEND_DIR) run build
	@bucket=$$(terraform -chdir=$(TF_DIR) output -raw bucket_name); \
	aws s3 sync $(DIST_DIR)/ s3://$$bucket/ --delete --region $(AWS_REGION)
	$(MAKE) invalidate

invalidate: _require-tf ## Invalidate the CloudFront cache
	@dist=$$(terraform -chdir=$(TF_DIR) output -raw cloudfront_distribution_id); \
	aws cloudfront create-invalidation --distribution-id $$dist --paths '/*' >/dev/null; \
	echo "CloudFront invalidation requested for $$dist"

down: _require-tf ## Destroy all AWS resources (bucket included) and stop containers
	terraform -chdir=$(TF_DIR) init -input=false
	terraform -chdir=$(TF_DIR) destroy -auto-approve -input=false
	-$(COMPOSE) down
	@echo "AWS resources destroyed, local containers stopped. Cost is back to \$0."

cost: ## Show AWS spend for the last 30 days
	@command -v aws >/dev/null || { echo "awscli not installed — see https://aws.amazon.com/cli/"; exit 1; }
	@start=$$(date -u -d '30 days ago' +%Y-%m-%d 2>/dev/null || date -u -v-30d +%Y-%m-%d); \
	end=$$(date -u +%Y-%m-%d); \
	aws ce get-cost-and-usage \
		--time-period Start=$$start,End=$$end \
		--granularity MONTHLY \
		--metrics UnblendedCost \
		--query 'ResultsByTime[].{Period: TimePeriod.Start, USD: Total.UnblendedCost.Amount}' \
		--output table

# --- housekeeping -----------------------------------------------------------

clean: ## Remove generated caches and the frontend build (leaves data/ alone)
	rm -rf $(DIST_DIR) .pytest_cache .ruff_cache
	find pipeline -name __pycache__ -type d -prune -exec rm -rf {} +

_require-tf:
	@command -v terraform >/dev/null || { echo "terraform not installed — see https://developer.hashicorp.com/terraform/install"; exit 1; }
	@test -d $(TF_DIR) || { echo "$(TF_DIR) not found — Phase 5 (AWS deploy) not implemented yet"; exit 1; }

_require-deploy: _require-tf
	@command -v npm >/dev/null || { echo "npm not installed — see https://nodejs.org"; exit 1; }
	@test -f $(FRONTEND_DIR)/package.json || { echo "$(FRONTEND_DIR)/package.json not found — Phase 4 not implemented yet"; exit 1; }
	@command -v aws >/dev/null || { echo "awscli not installed — see https://aws.amazon.com/cli/"; exit 1; }
