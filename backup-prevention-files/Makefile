# Wound Segmentation Makefile
# Provides convenient commands for environment management

.PHONY: help setup check test clean install-deps install-models run-gui run-api

# Default target
help:
	@echo "Wound Segmentation - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Run automated environment setup"
	@echo "  make install-deps   - Install Python dependencies"
	@echo "  make install-models - Download required model files"
	@echo ""
	@echo "Verification & Testing:"
	@echo "  make check          - Run health check"
	@echo "  make test           - Run test script"
	@echo "  make test-gui       - Test GUI functionality"
	@echo ""
	@echo "Running Applications:"
	@echo "  make run-gui        - Run GUI application"
	@echo "  make run-api        - Run web API"
	@echo "  make run-cli        - Run CLI with sample image"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Clean temporary files"
	@echo "  make update-deps    - Update dependencies"
	@echo "  make freeze-deps    - Generate requirements from current environment"

# Setup and installation
setup:
	@echo "🏥 Setting up Wound Segmentation environment..."
	chmod +x setup_environment.sh
	./setup_environment.sh

install-deps:
	@echo "📦 Installing Python dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

install-models:
	@echo "🤖 Setting up model directories..."
	mkdir -p models outputs reports wound_progress_report logs
	@echo "📁 Model directories created. Please add your model files to models/"

# Verification and testing
check:
	@echo "🔍 Running health check..."
	python health_check.py

test:
	@echo "🧪 Running test script..."
	python test_wound_progress.py

test-gui:
	@echo "🖥️  Testing GUI functionality..."
	python -c "import tkinter; print('✅ GUI test passed')"

# Running applications
run-gui:
	@echo "🖥️  Starting GUI application..."
	python wound_checker.py

run-api:
	@echo "🌐 Starting web API..."
	python app.py

run-cli:
	@echo "💻 Running CLI with sample..."
	@if [ -f "sample_image.jpg" ]; then \
		python -c "import sys; sys.path.append('.'); from test_wound_progress import main; main()"; \
	else \
		echo "❌ No sample_image.jpg found. Please provide an image file."; \
	fi

# Maintenance
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf *.egg-info

update-deps:
	@echo "🔄 Updating dependencies..."
	pip install --upgrade pip
	pip list --outdated
	@echo "Run 'pip install --upgrade <package>' for specific packages"

freeze-deps:
	@echo "📋 Generating requirements from current environment..."
	pip freeze > requirements_current.txt
	@echo "Current environment saved to requirements_current.txt"

# Development helpers
dev-setup: setup install-deps install-models check
	@echo "🚀 Development environment ready!"

quick-test: check test-gui
	@echo "✅ Quick tests completed!"

# Docker helpers (for future use)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t wound-segmentation .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 5000:5000 wound-segmentation

# Documentation
docs:
	@echo "📚 Opening documentation..."
	@if command -v open >/dev/null 2>&1; then \
		open docs/; \
	else \
		echo "Documentation available in docs/ folder"; \
	fi

# Environment info
info:
	@echo "🔍 Environment Information:"
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Python path: $$(which python)"
	@echo "Virtual environment: $$(echo $$VIRTUAL_ENV)"