#!/bin/bash

# Simple restore script for wound segmentation prevention files
# Run this after resetting your git branch

echo "🔄 Restoring prevention files..."

# Go to parent directory (project root)
cd ..

# Copy all files back
echo "📦 Restoring core prevention files..."
cp backup-prevention-files/requirements.txt ./
cp backup-prevention-files/setup_environment.sh ./
cp backup-prevention-files/health_check.py ./
cp backup-prevention-files/Makefile ./
cp backup-prevention-files/env.example ./

echo "📚 Restoring documentation files..."
cp backup-prevention-files/SETUP_GUIDE.md ./
cp backup-prevention-files/CONFIGURATION_PREVENTION.md ./
cp backup-prevention-files/README.md ./

# Make scripts executable
chmod +x setup_environment.sh
chmod +x health_check.py

echo "✅ Files restored successfully!"
echo "🧪 Run 'make check' to verify everything works"