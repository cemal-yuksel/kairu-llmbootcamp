"""
Launch Script for Academic Research Assistant v2.0
Enhanced startup script with environment checking and optimization
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import time

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'langchain',
        'openai',
        'chromadb',
        'sentence-transformers',
        'PyPDF2',
        'python-dotenv',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return missing_packages

def check_environment():
    """Check environment configuration"""
    print("\n🔍 Environment Check:")
    
    # Check .env file
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file found")
        
        # Check for required environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print(f"✅ OpenAI API Key: ***{openai_key[-4:]}")
        else:
            print("⚠️ OpenAI API Key not found")
            print("   Please add OPENAI_API_KEY to your .env file")
    else:
        print("⚠️ .env file not found")
        print("   Creating template .env file...")
        
        with open(".env", "w") as f:
            f.write("""# Academic Research Assistant v2.0 Configuration
# Add your API keys here

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3

# Optional: Other API configurations
# ANTHROPIC_API_KEY=your_anthropic_key_here
# GOOGLE_API_KEY=your_google_key_here

# Application Settings
DEBUG=false
LOG_LEVEL=info
MAX_FILE_SIZE=50
""")
        print("   Template .env file created. Please add your API keys.")

def install_missing_packages(packages):
    """Install missing packages"""
    if packages:
        print(f"\n📦 Installing missing packages: {', '.join(packages)}")
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
        
        print("✅ All packages installed successfully!")
    
    return True

def optimize_streamlit():
    """Set up Streamlit optimizations"""
    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(exist_ok=True)
    
    # Create optimized Streamlit config
    config_content = """
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 50

[browser]
gatherUsageStats = false
showErrorDetails = true

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
caching = true
displayEnabled = true

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true
"""
    
    config_path = config_dir / "config.toml"
    with open(config_path, "w") as f:
        f.write(config_content.strip())
    
    print("✅ Streamlit configuration optimized")

def check_system_resources():
    """Check system resources"""
    print("\n💻 System Resources:")
    
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"🔧 CPU: {cpu_count} cores, {cpu_usage}% usage")
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_usage = memory.percent
        print(f"💾 Memory: {memory_gb:.1f} GB total, {memory_usage}% used")
        
        # Disk info
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        print(f"💿 Disk: {disk_free_gb:.1f} GB free")
        
        # Warnings
        if memory_usage > 80:
            print("⚠️ High memory usage detected. Consider closing other applications.")
        
        if disk_free_gb < 1:
            print("⚠️ Low disk space. Please free up space for optimal performance.")
            
    except ImportError:
        print("📊 Resource monitoring not available (psutil not installed)")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/chroma_db",
        "data/pdfs",
        "data/sessions",
        "logs",
        "temp"
    ]
    
    print("\n📁 Creating directories:")
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def launch_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Academic Research Assistant v2.0...")
    print("="*60)
    print("🌐 Web interface will open in your default browser")
    print("📱 The application is mobile-responsive")
    print("⚡ Real-time streaming and interactive features enabled")
    print("🔄 Auto-refresh and session management active")
    print("="*60)
    
    # Change to UI directory
    ui_dir = Path("ui")
    if ui_dir.exists():
        os.chdir(ui_dir)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--browser.serverAddress", "localhost",
            "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("Try running manually: streamlit run ui/streamlit_app.py")

def show_startup_banner():
    """Display startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🚀 Academic Research Assistant v2.0                  ║
    ║                                                              ║
    ║        AI-Powered Research Analysis Platform                 ║
    ║        Built with LangChain, OpenAI & Streamlit            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Features:
    • 📄 Advanced document processing with multi-chain analysis
    • 🧠 Intelligent memory management and context retention
    • 🔍 Real-time literature search and citation management
    • 💬 Interactive research chat with streaming responses
    • 📊 Analytics dashboard with research insights
    • 🗂️ Project management and cross-project connections
    
    """
    print(banner)

def main():
    """Main launcher function"""
    show_startup_banner()
    
    print("🔧 System Setup:")
    print("="*40)
    
    # Check dependencies
    print("\n📋 Dependency Check:")
    missing = check_dependencies()
    
    if missing:
        install_choice = input(f"\nInstall missing packages? ({', '.join(missing)}) [y/N]: ")
        if install_choice.lower() in ['y', 'yes']:
            if not install_missing_packages(missing):
                print("❌ Failed to install dependencies. Please install manually.")
                return
        else:
            print("⚠️ Some features may not work without required packages.")
    
    # Check environment
    check_environment()
    
    # Create directories
    create_directories()
    
    # Optimize Streamlit
    optimize_streamlit()
    
    # System resources
    check_system_resources()
    
    # Launch application
    print(f"\n✅ Setup complete!")
    time.sleep(1)
    
    launch_choice = input("\n🚀 Launch the application now? [Y/n]: ")
    if launch_choice.lower() not in ['n', 'no']:
        launch_streamlit()
    else:
        print("\n📝 To launch manually, run:")
        print("   streamlit run ui/streamlit_app.py")
        print("\n🌐 Or use the web interface:")
        print("   http://localhost:8501")

if __name__ == "__main__":
    main()