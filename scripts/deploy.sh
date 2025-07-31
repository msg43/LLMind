#!/bin/bash

# LLMind Deployment Script
# Automated deployment for Apple Silicon environments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="LLMind"
CONTAINER_NAME="llmind-app"
COMPOSE_FILE="docker-compose.yml"
BACKUP_DIR="./backups"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if running on macOS (for Apple Silicon optimization)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Running on macOS - Apple Silicon optimizations enabled"
    else
        log_warning "Not running on macOS - some optimizations may not be available"
    fi

    log_success "System requirements check completed"
}

backup_data() {
    log_info "Creating backup of existing data..."

    mkdir -p "$BACKUP_DIR"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_file="$BACKUP_DIR/llmind_backup_$timestamp.tar.gz"

    if docker volume ls | grep -q llmind; then
        log_info "Backing up Docker volumes..."

        # Create temporary container to access volumes
                docker run --rm -v llmind_models:/models \
            -v llmind_documents:/documents \
            -v llmind_vector_store:/vector_store \
                   -v "$(pwd)/$BACKUP_DIR":/backup \
                   alpine:latest \
                   tar czf "/backup/volumes_$timestamp.tar.gz" /models /documents /vector_store

        log_success "Backup created: $backup_file"
    else
        log_info "No existing volumes found to backup"
    fi
}

deploy() {
    log_info "Starting $APP_NAME deployment..."

    # Build and start containers
    log_info "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache

    log_info "Starting containers..."
    docker-compose -f "$COMPOSE_FILE" up -d

    # Wait for health check
    log_info "Waiting for application to be healthy..."
    timeout=120
    counter=0

    while [ $counter -lt $timeout ]; do
        if docker-compose -f "$COMPOSE_FILE" ps | grep -q "healthy"; then
            log_success "$APP_NAME is running and healthy!"
            break
        fi

        if [ $counter -eq $((timeout-10)) ]; then
            log_warning "Application is taking longer than expected to start..."
        fi

        sleep 2
        counter=$((counter+2))
    done

    if [ $counter -ge $timeout ]; then
        log_error "Application failed to start within $timeout seconds"
        log_info "Checking container logs..."
        docker-compose -f "$COMPOSE_FILE" logs --tail=50
        exit 1
    fi

    # Display deployment info
    echo
    log_success "🚀 $APP_NAME deployed successfully!"
    echo
    log_info "📊 Access the application at: http://localhost:8000"
    log_info "🏥 Health check: http://localhost:8000/api/health"
    log_info "📋 API docs: http://localhost:8000/docs"
    echo
}

stop() {
    log_info "Stopping $APP_NAME..."
    docker-compose -f "$COMPOSE_FILE" down
    log_success "$APP_NAME stopped"
}

restart() {
    log_info "Restarting $APP_NAME..."
    stop
    deploy
}

update() {
    log_info "Updating $APP_NAME..."

    # Pull latest code (if in git repo)
    if [ -d ".git" ]; then
        log_info "Pulling latest code..."
        git pull
    fi

    # Backup before update
    backup_data

    # Rebuild and restart
    log_info "Rebuilding application..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    docker-compose -f "$COMPOSE_FILE" up -d

    log_success "Update completed"
}

logs() {
    docker-compose -f "$COMPOSE_FILE" logs -f
}

status() {
    log_info "$APP_NAME Status:"
    docker-compose -f "$COMPOSE_FILE" ps

    echo
    log_info "Docker Volumes:"
    docker volume ls | grep llmind || log_info "No volumes found"

    echo
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" $(docker-compose -f "$COMPOSE_FILE" ps -q) 2>/dev/null || log_info "No running containers"
}

cleanup() {
    log_warning "This will remove all containers, volumes, and images. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Cleaning up $APP_NAME..."

        # Stop and remove containers
        docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans

        # Remove images
        docker image prune -a -f --filter label=maintainer="LLMind Team"

        # Remove volumes (optional)
        docker volume ls | grep llmind | awk '{print $2}' | xargs -r docker volume rm

        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

show_help() {
    echo "LLMind Deployment Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  deploy     Deploy the application"
    echo "  stop       Stop the application"
    echo "  restart    Restart the application"
    echo "  update     Update and redeploy the application"
    echo "  logs       Show application logs"
    echo "  status     Show application status"
    echo "  backup     Create a backup of application data"
    echo "  cleanup    Remove all containers, volumes, and images"
    echo "  help       Show this help message"
    echo
    echo "Examples:"
    echo "  $0 deploy          # Deploy LLMind"
    echo "  $0 logs            # View real-time logs"
    echo "  $0 status          # Check application status"
}

# Main script logic
case "${1:-}" in
    deploy)
        check_requirements
        deploy
        ;;
    stop)
        stop
        ;;
    restart)
        check_requirements
        restart
        ;;
    update)
        check_requirements
        update
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    backup)
        backup_data
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: ${1:-}"
        echo
        show_help
        exit 1
        ;;
esac
