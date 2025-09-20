#!/bin/bash

# Backup script for wound segmentation prevention files
# Run this before resetting your git branch

set -e

BACKUP_DIR="$HOME/Desktop/wound-seg-backup-$(date +%Y%m%d-%H%M%S)"

echo "🏥 Creating backup of prevention files..."
echo "📁 Backup directory: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Core prevention files
echo "📦 Backing up core prevention files..."
cp requirements.txt "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  requirements.txt not found"
cp setup_environment.sh "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  setup_environment.sh not found"
cp health_check.py "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  health_check.py not found"
cp Makefile "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  Makefile not found"
cp env.example "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  env.example not found"

# Documentation files
echo "📚 Backing up documentation files..."
cp SETUP_GUIDE.md "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  SETUP_GUIDE.md not found"
cp CONFIGURATION_PREVENTION.md "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  CONFIGURATION_PREVENTION.md not found"
cp README.md "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  README.md not found"

# Make scripts executable
chmod +x "$BACKUP_DIR"/*.sh 2>/dev/null || true
chmod +x "$BACKUP_DIR"/*.py 2>/dev/null || true

# Create restore script
cat > "$BACKUP_DIR/restore_files.sh" << 'EOF'
#!/bin/bash

# Restore script for wound segmentation prevention files
# Run this after resetting your git branch

set -e

BACKUP_DIR="$(dirname "$0")"

echo "🔄 Restoring prevention files from: $BACKUP_DIR"

# Core prevention files
echo "📦 Restoring core prevention files..."
cp "$BACKUP_DIR/requirements.txt" ./ 2>/dev/null || echo "⚠️  requirements.txt not found in backup"
cp "$BACKUP_DIR/setup_environment.sh" ./ 2>/dev/null || echo "⚠️  setup_environment.sh not found in backup"
cp "$BACKUP_DIR/health_check.py" ./ 2>/dev/null || echo "⚠️  health_check.py not found in backup"
cp "$BACKUP_DIR/Makefile" ./ 2>/dev/null || echo "⚠️  Makefile not found in backup"
cp "$BACKUP_DIR/env.example" ./ 2>/dev/null || echo "⚠️  env.example not found in backup"

# Documentation files
echo "📚 Restoring documentation files..."
cp "$BACKUP_DIR/SETUP_GUIDE.md" ./ 2>/dev/null || echo "⚠️  SETUP_GUIDE.md not found in backup"
cp "$BACKUP_DIR/CONFIGURATION_PREVENTION.md" ./ 2>/dev/null || echo "⚠️  CONFIGURATION_PREVENTION.md not found in backup"
cp "$BACKUP_DIR/README.md" ./ 2>/dev/null || echo "⚠️  README.md not found in backup"

# Make scripts executable
chmod +x setup_environment.sh 2>/dev/null || true
chmod +x health_check.py 2>/dev/null || true

echo "✅ Files restored successfully!"
echo "🧪 Run 'make check' to verify everything works"
EOF

chmod +x "$BACKUP_DIR/restore_files.sh"

echo "✅ Backup completed successfully!"
echo "📁 Backup location: $BACKUP_DIR"
echo ""
echo "🔄 To restore after git reset:"
echo "   1. cd /path/to/wound-segmentation"
echo "   2. $BACKUP_DIR/restore_files.sh"
echo "   3. make check"
echo ""
echo "📋 Files backed up:"
ls -la "$BACKUP_DIR"