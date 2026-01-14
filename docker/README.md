# StreamHub Docker Configuration

This directory contains all Docker configurations for running StreamHub in both development and production environments.

## Quick Start

### Development

```bash
# Copy environment file and configure
cp .env.example .env

# Start all services
make dev

# Or with docker-compose directly
docker-compose up -d
```

**Access Points:**
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Frontend:** http://localhost:3000

### Production

```bash
# Copy and configure production environment
cp .env.prod.example .env.prod

# Start production services
make prod

# Or with docker-compose directly
docker-compose -f docker-compose.prod.yml up -d
```

## Directory Structure

```
docker/
├── Dockerfile.api           # FastAPI backend (multi-stage)
├── Dockerfile.frontend      # Next.js frontend (multi-stage)
├── docker-compose.yml       # Development configuration
├── docker-compose.prod.yml  # Production configuration
├── .env.example             # Development environment template
├── .env.prod.example        # Production environment template
├── .dockerignore.api        # API build exclusions
├── .dockerignore.frontend   # Frontend build exclusions
├── Makefile                 # Convenience commands
├── init-scripts/
│   └── postgres/
│       └── 01-init.sql      # Database initialization
└── README.md                # This file
```

## Services

| Service | Description | Dev Port | Prod Port |
|---------|-------------|----------|----------|
| `api` | FastAPI backend | 8000 | - (via Traefik) |
| `worker` | Content processor | - | - |
| `frontend` | Next.js dashboard | 3000 | - (via Traefik) |
| `postgres` | PostgreSQL + pgvector | 5432 | - (internal) |
| `redis` | Cache & queue | 6379 | - (internal) |
| `traefik` | Reverse proxy (prod) | - | 80, 443 |

## Make Commands

```bash
make help          # Show all commands
make dev           # Start development
make dev-build     # Build and start development
make dev-tools     # Start with pgAdmin & Redis Commander
make prod          # Start production
make down          # Stop all services
make logs          # View all logs
make logs-api      # View API logs
make shell-api     # Shell into API container
make shell-postgres # PostgreSQL CLI
make db-upgrade    # Run database migrations
make test          # Run tests
make clean         # Remove all containers and volumes
```

## Docker Images

### API Image Targets

- `production`: Optimized runtime with minimal footprint
- `development`: Includes dev tools, hot reload enabled

### Frontend Image Targets

- `runner`: Standalone production build
- `development`: Development mode with hot reload

## Environment Variables

See `.env.example` for development and `.env.prod.example` for production variables.

**Critical Production Variables:**
- `SECRET_KEY`: Strong random string
- `POSTGRES_PASSWORD`: Database password
- `REDIS_PASSWORD`: Redis password
- `OPENAI_API_KEY`: OpenAI API credentials
- `DOMAIN`: Production domain name
- `ACME_EMAIL`: Let's Encrypt email

## Health Checks

All services include health checks:

```bash
# Check service health
make health

# Manual check
curl http://localhost:8000/health
```

## Resource Limits

| Service | Dev Memory | Prod Memory |
|---------|------------|-------------|
| API | 1GB | 2GB |
| Worker | 1GB | 2GB |
| Frontend | 2GB | 1GB |
| PostgreSQL | 1GB | 4GB |
| Redis | 512MB | 2GB |

## Networking

- **Development:** Single `streamhub-network` bridge network
- **Production:** 
  - `streamhub-internal`: Internal service communication (no external access)
  - `streamhub-proxy`: External access via Traefik

## Volumes

- `postgres_data`: PostgreSQL data persistence
- `redis_data`: Redis persistence (AOF enabled)
- `traefik_certs`: SSL certificates (production)

## Security

Production containers include:
- Non-root users
- Read-only filesystems
- No new privileges
- Resource limits
- Internal-only networking for databases
- SSL/TLS via Traefik

## Troubleshooting

### Container won't start
```bash
docker-compose logs <service>
make logs-api
```

### Database connection issues
```bash
# Check if postgres is healthy
docker-compose ps postgres

# Test connection
make shell-postgres
```

### Reset everything
```bash
make clean
make dev-build
```
