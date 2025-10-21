#!/bin/bash

# Academic Research Assistant v2.0 - Unix Launcher
# Enhanced startup script for Linux/macOS systems

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${1}${2}${NC}"
}

print_banner() {
    print_color $BLUE "============================================================"
    print_color $CYAN "  🚀 Academic Research Assistant v2.0 - Unix Launcher"
    print_color $BLUE "============================================================"
    echo
}

check_python() {
    print_color $YELLOW "🐍 Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        print_color $GREEN "✅ Python3 found: $(python3 --version)"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        print_color $GREEN "✅ Python found: $(python --version)"
    else
        print_color $RED "❌ Python not found!"
        print_color $RED "Please install Python 3.8+ from https://python.org"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 8 ]); then
        print_color $RED "❌ Python version $PYTHON_VERSION is too old"
        print_color $RED "Please upgrade to Python 3.8 or higher"
        exit 1
    fi
    
    echo
}

check_venv() {
    print_color $YELLOW "🏠 Checking virtual environment..."
    
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_color $GREEN "✅ Virtual Environment: Active ($VIRTUAL_ENV)"
    else
        print_color $YELLOW "⚠️ Virtual Environment: Not detected"
        
        # Check if there's a virtual environment in the project
        if [ -d "venv" ] || [ -d ".venv" ] || [ -d "env" ]; then
            print_color $CYAN "💡 Virtual environment found in project directory"
            read -p "Activate virtual environment? (y/N): " activate_venv
            
            if [[ $activate_venv =~ ^[Yy]$ ]]; then
                if [ -d "venv" ]; then
                    source venv/bin/activate
                elif [ -d ".venv" ]; then
                    source .venv/bin/activate
                elif [ -d "env" ]; then
                    source env/bin/activate
                fi
                print_color $GREEN "✅ Virtual environment activated"
            fi
        else
            print_color $CYAN "💡 Consider using a virtual environment for better dependency management"
        fi
    fi
    echo
}

check_system_info() {
    print_color $YELLOW "💻 System Information:"
    echo "OS: $(uname -s)"
    echo "Architecture: $(uname -m)"
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY=$(free -h | grep '^Mem:' | awk '{print $2}')
        print_color $CYAN "Memory: $MEMORY"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        MEMORY=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2 $3}')
        print_color $CYAN "Memory: $MEMORY"
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    print_color $CYAN "Available Disk Space: $DISK_SPACE"
    
    echo
}

make_executable() {
    # Make the script executable if it isn't already
    if [ ! -x "$0" ]; then
        print_color $YELLOW "🔧 Making script executable..."
        chmod +x "$0"
        print_color $GREEN "✅ Script is now executable"
    fi
}

check_port() {
    print_color $YELLOW "🌐 Checking port 8501..."
    
    if command -v lsof &> /dev/null; then
        if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
            print_color $YELLOW "⚠️ Port 8501 is already in use"
            read -p "Kill existing process and continue? (y/N): " kill_process
            
            if [[ $kill_process =~ ^[Yy]$ ]]; then
                print_color $CYAN "🔄 Killing existing process..."
                lsof -ti:8501 | xargs kill -9 2>/dev/null || true
                sleep 2
                print_color $GREEN "✅ Port cleared"
            else
                print_color $RED "❌ Cannot proceed with port 8501 occupied"
                exit 1
            fi
        else
            print_color $GREEN "✅ Port 8501 is available"
        fi
    fi
    echo
}

create_desktop_shortcut() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        DESKTOP_DIR="$HOME/Desktop"
        if [ -d "$DESKTOP_DIR" ]; then
            read -p "Create desktop shortcut? (y/N): " create_shortcut
            
            if [[ $create_shortcut =~ ^[Yy]$ ]]; then
                SHORTCUT_FILE="$DESKTOP_DIR/Academic Research Assistant.desktop"
                cat > "$SHORTCUT_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Academic Research Assistant v2.0
Comment=AI-Powered Research Analysis Platform
Exec=bash $(pwd)/start.sh
Icon=applications-science
Terminal=true
Categories=Education;Science;
EOF
                chmod +x "$SHORTCUT_FILE"
                print_color $GREEN "✅ Desktop shortcut created"
            fi
        fi
    fi
}

main() {
    print_banner
    
    # System checks
    check_python
    check_venv
    check_system_info
    make_executable
    check_port
    
    print_color $BLUE "🚀 Starting Academic Research Assistant v2.0..."
    print_color $BLUE "=================================================="
    echo
    
    # Run the Python launcher
    if ! $PYTHON_CMD launch.py; then
        print_color $RED "❌ Application failed to start"
        print_color $CYAN "💡 Try running manually: streamlit run ui/streamlit_app.py"
        exit 1
    fi
    
    print_color $GREEN "👋 Application finished successfully"
    
    # Offer to create desktop shortcut
    create_desktop_shortcut
}

# Trap to handle Ctrl+C gracefully
trap 'print_color $YELLOW "\n🛑 Application interrupted by user"; exit 0' INT

# Run main function
main "$@"