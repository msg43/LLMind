#!/bin/bash

# LLMind Utility Script
# ======================
# This script provides common maintenance tasks for LLMind

set -e

# Function to show help
show_help() {
    echo "LLMind Utility Script"
    echo "========================"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  clean         Clean temporary files and caches"
    echo "  backup        Create a backup of data directory"
    echo "  restore       Restore from a backup"
    echo "  reset         Reset vector store and models (keeps documents)"
    echo "  purge         Complete data reset (WARNING: deletes everything)"
    echo "  status        Show application status and disk usage"
    echo "  logs          Show recent application logs"
    echo "  health        Run health checks"
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 clean              # Clean temporary files"
    echo "  $0 backup             # Create backup"
    echo "  $0 status             # Show status"
    echo "  $0 reset              # Reset vector store"
}

# Function to clean temporary files
clean() {
    echo "🧹 Cleaning temporary files and caches..."

    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true

    # Remove test artifacts
    rm -rf test_output/
    rm -rf .pytest_cache/
    rm -rf htmlcov/
    rm -rf .coverage

    # Clean logs older than 7 days
    if [ -d "logs" ]; then
        find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
    fi

    # Clean data cache
    if [ -d "data/cache" ]; then
        rm -rf data/cache/*
    fi

    # Clean audio files older than 1 day
    if [ -d "data/audio" ]; then
        find data/audio/ -name "*.wav" -mtime +1 -delete 2>/dev/null || true
    fi

    echo "✅ Cleanup completed"
}

# Function to create backup
backup() {
    echo "💾 Creating backup..."

    BACKUP_DIR="data/backups"
    BACKUP_NAME="llmind_backup_$(date +%Y%m%d_%H%M%S)"
    BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME.tar.gz"

    mkdir -p "$BACKUP_DIR"

    # Create backup excluding temporary files
    tar -czf "$BACKUP_PATH" \
        --exclude="data/backups" \
        --exclude="data/cache" \
        --exclude="data/audio/*.wav" \
        --exclude="logs/*.log" \
        data/ 2>/dev/null || true

    if [ -f "$BACKUP_PATH" ]; then
        BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
        echo "✅ Backup created: $BACKUP_PATH ($BACKUP_SIZE)"

        # Clean old backups (keep last 5)
        cd "$BACKUP_DIR"
        ls -t llmind_backup_*.tar.gz | tail -n +6 | xargs rm -f 2>/dev/null || true
        cd - >/dev/null

        echo "🗂️  Old backups cleaned (keeping last 5)"
    else
        echo "❌ Backup failed"
        exit 1
    fi
}

# Function to restore from backup
restore() {
    echo "🔄 Restoring from backup..."

    BACKUP_DIR="data/backups"

    if [ ! -d "$BACKUP_DIR" ]; then
        echo "❌ No backups directory found"
        exit 1
    fi

    # Find latest backup
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/llmind_backup_*.tar.gz 2>/dev/null | head -n 1)

    if [ -z "$LATEST_BACKUP" ]; then
        echo "❌ No backups found"
        exit 1
    fi

    echo "📋 Latest backup: $(basename "$LATEST_BACKUP")"
    read -p "Restore from this backup? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Stop application if running
        ./scripts/stop.sh 2>/dev/null || true

        # Backup current data
        TEMP_BACKUP="data_backup_$(date +%s)"
        mv data "$TEMP_BACKUP" 2>/dev/null || true

        # Restore from backup
        tar -xzf "$LATEST_BACKUP"

        if [ $? -eq 0 ]; then
            echo "✅ Restore completed successfully"
            rm -rf "$TEMP_BACKUP" 2>/dev/null || true
        else
            echo "❌ Restore failed, reverting..."
            rm -rf data 2>/dev/null || true
            mv "$TEMP_BACKUP" data 2>/dev/null || true
            exit 1
        fi
    else
        echo "ℹ️  Restore cancelled"
    fi
}

# Function to reset vector store and models
reset() {
    echo "🔄 Resetting vector store and models..."

    read -p "This will delete vector store and downloaded models. Continue? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./scripts/stop.sh 2>/dev/null || true

        rm -rf data/vector_store/*
        rm -rf data/models/*
        rm -rf data/cache/*

        echo "✅ Reset completed"
        echo "ℹ️  Documents are preserved in data/documents/"
    else
        echo "ℹ️  Reset cancelled"
    fi
}

# Function to purge all data
purge() {
    echo "⚠️  DANGER: Complete data purge"
    echo "This will delete ALL data including documents, models, and vector store."
    echo ""
    read -p "Type 'DELETE' to confirm: " confirm

    if [ "$confirm" = "DELETE" ]; then
        ./scripts/stop.sh 2>/dev/null || true

        rm -rf data/documents/*
        rm -rf data/vector_store/*
        rm -rf data/models/*
        rm -rf data/audio/*
        rm -rf data/cache/*
        rm -rf logs/*

        echo "💥 All data purged"
    else
        echo "ℹ️  Purge cancelled"
    fi
}

# Function to show status
status() {
    echo "📊 LLMind Status"
    echo "=================="
    echo ""

    # Check if application is running
    PORT=${PORT:-8000}
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
        echo "🟢 Status: Running (PID: $PID, Port: $PORT)"
    else
        echo "🔴 Status: Stopped"
    fi

    echo ""
    echo "💾 Disk Usage:"
    if [ -d "data" ]; then
        du -sh data/* 2>/dev/null | sort -hr || echo "  No data directories found"
    else
        echo "  No data directory found"
    fi

    echo ""
    echo "📁 File Counts:"
    [ -d "data/documents" ] && echo "  Documents: $(find data/documents -type f 2>/dev/null | wc -l)"
    [ -d "data/models" ] && echo "  Models: $(find data/models -type f -name "*.safetensors" -o -name "*.bin" 2>/dev/null | wc -l)"
    [ -d "data/vector_store" ] && echo "  Vector indexes: $(find data/vector_store -name "*.faiss" 2>/dev/null | wc -l)"

    echo ""
    echo "🕐 Recent Activity:"
    if [ -d "logs" ] && [ -n "$(ls logs/*.log 2>/dev/null)" ]; then
        echo "  Last log entry: $(tail -n 1 logs/*.log 2>/dev/null | head -n 1 | cut -c1-80)..."
    else
        echo "  No recent logs found"
    fi
}

# Function to show logs
logs() {
    echo "📜 Recent Application Logs"
    echo "========================="
    echo ""

    if [ -d "logs" ] && [ -n "$(ls logs/*.log 2>/dev/null)" ]; then
        tail -n 50 logs/*.log 2>/dev/null | head -n 100
    else
        echo "No log files found"
    fi
}

# Function to run health checks
health() {
    echo "🔍 Running Health Checks"
    echo "======================="
    echo ""

    # Check virtual environment
    if [ -d "venv" ]; then
        echo "✅ Virtual environment: Found"
    else
        echo "❌ Virtual environment: Missing"
    fi

    # Check required files
    for file in main.py config.py requirements.txt; do
        if [ -f "$file" ]; then
            echo "✅ $file: Found"
        else
            echo "❌ $file: Missing"
        fi
    done

    # Check data directories
    for dir in data/documents data/vector_store data/models data/audio; do
        if [ -d "$dir" ]; then
            echo "✅ $dir: Found"
        else
            echo "⚠️  $dir: Missing (will be created on startup)"
        fi
    done

    # Check port availability
    PORT=${PORT:-8000}
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $PORT: In use"
    else
        echo "✅ Port $PORT: Available"
    fi

    echo ""
    echo "🏥 Health check completed"
}

# Main script logic
case "${1:-}" in
    clean)
        clean
        ;;
    backup)
        backup
        ;;
    restore)
        restore
        ;;
    reset)
        reset
        ;;
    purge)
        purge
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    health)
        health
        ;;
    --help|-h|help)
        show_help
        ;;
    *)
        echo "❌ Unknown command: ${1:-}"
        echo ""
        show_help
        exit 1
        ;;
esac
