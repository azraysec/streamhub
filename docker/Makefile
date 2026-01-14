# =============================================================================
# StreamHub - Docker Makefile
# =============================================================================
# Convenience commands for Docker operations
#
# Usage:
#   make dev          - Start development environment
#   make prod         - Start production environment
#   make down         - Stop all services
#   make logs         - View logs
#   make help         - Show all available commands
# =============================================================================

.PHONY: help dev dev-build dev-tools prod prod-build down logs clean \
        shell-api shell-frontend shell-postgres shell-redis \
        db-migrate db-upgrade db-downgrade test lint

# Default target
.DEFAULT_GOAL := help

# Colors for output
COLOR_RESET   = \033[0m
COLOR_INFO    = \033[36m
COLOR_SUCCESS = \033[32m
COLOR_WARNING = \033[33m

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "$(COLOR_INFO)StreamHub Docker Commands$(COLOR_RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_SUCCESS)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Development
# =============================================================================
dev: ## Start development environment
	@echo "$(COLOR_INFO)Starting development environment...$(COLOR_RESET)"
	docker-compose up -d
	@echo "$(COLOR_SUCCESS)Development environment started!$(COLOR_RESET)"
	@echo "  API:      http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo "  API Docs: http://localhost:8000/docs"

dev-build: ## Build and start development environment
	@echo "$(COLOR_INFO)Building and starting development environment...$(COLOR_RESET)"
	docker-compose up -d --build

dev-tools: ## Start development environment with admin tools (pgAdmin, Redis Commander)
	@echo "$(COLOR_INFO)Starting development environment with tools...$(COLOR_RESET)"
	docker-compose --profile tools up -d
	@echo "  pgAdmin:         http://localhost:5050"
	@echo "  Redis Commander: http://localhost:8081"

# =============================================================================
# Production
# =============================================================================
prod: ## Start production environment
	@echo "$(COLOR_INFO)Starting production environment...$(COLOR_RESET)"
	docker-compose -f docker-compose.prod.yml up -d

prod-build: ## Build and start production environment
	@echo "$(COLOR_INFO)Building and starting production environment...$(COLOR_RESET)"
	docker-compose -f docker-compose.prod.yml up -d --build

prod-pull: ## Pull latest images and restart production
	@echo "$(COLOR_INFO)Pulling latest images...$(COLOR_RESET)"
	docker-compose -f docker-compose.prod.yml pull
	docker-compose -f docker-compose.prod.yml up -d

# =============================================================================
# Stopping Services
# =============================================================================
down: ## Stop all services
	@echo "$(COLOR_INFO)Stopping all services...$(COLOR_RESET)"
	docker-compose down
	docker-compose -f docker-compose.prod.yml down 2>/dev/null || true

down-v: ## Stop all services and remove volumes
	@echo "$(COLOR_WARNING)Stopping all services and removing volumes...$(COLOR_RESET)"
	docker-compose down -v
	docker-compose -f docker-compose.prod.yml down -v 2>/dev/null || true

restart: ## Restart all services
	@echo "$(COLOR_INFO)Restarting all services...$(COLOR_RESET)"
	docker-compose restart

# =============================================================================
# Logs
# =============================================================================
logs: ## View logs from all services
	docker-compose logs -f

logs-api: ## View API logs
	docker-compose logs -f api

logs-worker: ## View worker logs
	docker-compose logs -f worker

logs-frontend: ## View frontend logs
	docker-compose logs -f frontend

logs-postgres: ## View PostgreSQL logs
	docker-compose logs -f postgres

logs-redis: ## View Redis logs
	docker-compose logs -f redis

# =============================================================================
# Shell Access
# =============================================================================
shell-api: ## Open shell in API container
	docker-compose exec api /bin/bash

shell-frontend: ## Open shell in frontend container
	docker-compose exec frontend /bin/sh

shell-postgres: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U streamhub -d streamhub

shell-redis: ## Open Redis CLI
	docker-compose exec redis redis-cli

# =============================================================================
# Database Operations
# =============================================================================
db-migrate: ## Generate new database migration
	@read -p "Migration message: " msg; \
	docker-compose exec api alembic revision --autogenerate -m "$$msg"

db-upgrade: ## Apply database migrations
	@echo "$(COLOR_INFO)Applying database migrations...$(COLOR_RESET)"
	docker-compose exec api alembic upgrade head

db-downgrade: ## Rollback last database migration
	@echo "$(COLOR_WARNING)Rolling back last migration...$(COLOR_RESET)"
	docker-compose exec api alembic downgrade -1

db-history: ## Show migration history
	docker-compose exec api alembic history

db-current: ## Show current migration revision
	docker-compose exec api alembic current

# =============================================================================
# Testing & Quality
# =============================================================================
test: ## Run tests
	@echo "$(COLOR_INFO)Running tests...$(COLOR_RESET)"
	docker-compose exec api pytest -v

test-cov: ## Run tests with coverage
	@echo "$(COLOR_INFO)Running tests with coverage...$(COLOR_RESET)"
	docker-compose exec api pytest --cov=src --cov-report=html

lint: ## Run linting
	@echo "$(COLOR_INFO)Running linters...$(COLOR_RESET)"
	docker-compose exec api black --check src/
	docker-compose exec api isort --check-only src/
	docker-compose exec api flake8 src/

format: ## Format code
	@echo "$(COLOR_INFO)Formatting code...$(COLOR_RESET)"
	docker-compose exec api black src/
	docker-compose exec api isort src/

# =============================================================================
# Cleanup
# =============================================================================
clean: ## Remove all containers, networks, and volumes
	@echo "$(COLOR_WARNING)Cleaning up all Docker resources...$(COLOR_RESET)"
	docker-compose down -v --rmi local --remove-orphans
	docker-compose -f docker-compose.prod.yml down -v --rmi local --remove-orphans 2>/dev/null || true

prune: ## Remove unused Docker resources
	@echo "$(COLOR_WARNING)Pruning unused Docker resources...$(COLOR_RESET)"
	docker system prune -f
	docker volume prune -f

# =============================================================================
# Status
# =============================================================================
status: ## Show status of all services
	@echo "$(COLOR_INFO)Service Status:$(COLOR_RESET)"
	docker-compose ps

health: ## Check health of all services
	@echo "$(COLOR_INFO)Health Check:$(COLOR_RESET)"
	@docker-compose ps --format "table {{.Name}}\t{{.Status}}"

stats: ## Show resource usage statistics
	docker stats --no-stream $$(docker-compose ps -q)
