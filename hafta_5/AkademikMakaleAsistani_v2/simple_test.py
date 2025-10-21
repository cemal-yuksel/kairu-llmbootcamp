"""
Simple test version of the Academic Research Assistant v2.0
Minimal version to demonstrate the UI concepts without complex dependencies
"""

import os
import json
import datetime
from pathlib import Path

class SimpleApp:
    def __init__(self):
        self.session_id = f"simple_session_{int(datetime.datetime.now().timestamp())}"
        self.chat_history = []
        self.processed_documents = []
        
    def run(self):
        """Run simple CLI version"""
        print("🚀 Academic Research Assistant v2.0 - Simple Test Version")
        print("=" * 60)
        
        while True:
            print("\n📋 Menu:")
            print("1. 📄 Process Document (Simulated)")
            print("2. 💬 Ask Question (Simulated)")  
            print("3. 📊 Show Status")
            print("4. 💾 Save Session")
            print("5. 🚪 Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                self.process_document()
            elif choice == "2":
                self.ask_question()
            elif choice == "3":
                self.show_status()
            elif choice == "4":
                self.save_session()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def process_document(self):
        """Simulate document processing"""
        filename = input("📄 Enter document name (e.g., 'research_paper.pdf'): ").strip()
        
        if not filename:
            filename = "sample_document.pdf"
        
        print(f"🔄 Processing {filename}...")
        
        # Simulate processing
        import time
        time.sleep(2)
        
        doc_info = {
            'filename': filename,
            'processed_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'text_length': 15000,  # Simulated
            'analysis': {
                'research_field': 'Computer Science',
                'novelty_score': 8.5,
                'key_topics': ['AI', 'Machine Learning', 'Natural Language Processing']
            }
        }
        
        self.processed_documents.append(doc_info)
        print(f"✅ Successfully processed {filename}")
        print(f"📊 Analysis: {doc_info['analysis']['research_field']}, Novelty: {doc_info['analysis']['novelty_score']}/10")
    
    def ask_question(self):
        """Simulate question answering"""
        question = input("❓ Enter your research question: ").strip()
        
        if not question:
            question = "What are the main findings?"
        
        print(f"🤔 Processing question: {question}")
        
        # Simulate AI processing
        import time
        time.sleep(1.5)
        
        # Generate a simulated response
        responses = [
            "Based on the analyzed documents, the main findings suggest significant advances in AI methodology.",
            "The research indicates promising results in natural language processing applications.",
            "The documents reveal important insights about machine learning optimization techniques.",
            "Analysis shows notable improvements in computational efficiency and accuracy."
        ]
        
        import random
        answer = random.choice(responses)
        
        chat_entry = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'confidence': random.uniform(0.7, 0.95)
        }
        
        self.chat_history.append(chat_entry)
        
        print(f"🤖 Answer: {answer}")
        print(f"📊 Confidence: {chat_entry['confidence']:.2f}")
    
    def show_status(self):
        """Show current status"""
        print("\n📊 Academic Research Assistant Status")
        print("=" * 40)
        print(f"🆔 Session ID: {self.session_id}")
        print(f"📄 Documents Processed: {len(self.processed_documents)}")
        print(f"💬 Questions Asked: {len(self.chat_history)}")
        print(f"🕐 Session Start: {self.session_id.split('_')[2]}")
        
        if self.processed_documents:
            print("\n📚 Recent Documents:")
            for doc in self.processed_documents[-3:]:  # Show last 3
                print(f"  • {doc['filename']} ({doc['processed_date']})")
        
        if self.chat_history:
            print("\n💭 Recent Questions:")
            for chat in self.chat_history[-3:]:  # Show last 3
                print(f"  Q: {chat['question'][:50]}...")
                print(f"  A: {chat['answer'][:50]}...")
    
    def save_session(self):
        """Save session to file"""
        session_data = {
            'session_id': self.session_id,
            'chat_history': self.chat_history,
            'processed_documents': self.processed_documents,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        filename = f"data/session_{self.session_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Session saved to {filename}")

def check_environment():
    """Check basic environment"""
    print("🔍 Environment Check:")
    print("-" * 30)
    
    # Check Python version
    import sys
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check if .env exists
    if Path(".env").exists():
        print("✅ .env file found")
    else:
        print("⚠️ .env file not found")
    
    # Check directories
    dirs_to_check = ["data", "ui", "chains", "memory", "tools", "streaming"]
    for dir_name in dirs_to_check:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
    
    print()

def create_sample_env():
    """Create a sample .env file"""
    env_content = """# Academic Research Assistant v2.0 Configuration
# Simple test configuration

# OpenAI API Key (add your actual key here)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration  
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3

# Application Settings
DEBUG=true
LOG_LEVEL=info
MAX_FILE_SIZE=50

# Database Settings
CHROMA_DB_PATH=./data/chroma_db
"""
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        print("📝 Created sample .env file")

def main():
    """Main function"""
    print("🚀 Academic Research Assistant v2.0 - Test Launch")
    print("=" * 60)
    
    # Check environment
    check_environment()
    
    # Create sample .env if needed
    create_sample_env()
    
    # Create basic directories
    dirs = ["data", "logs", "temp"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created/verified {dir_name}/ directory")
    
    print("\n✅ Basic setup complete!")
    print("🎯 This is a simplified test version to demonstrate the system structure.")
    print("📦 For full functionality, install all dependencies from requirements.txt")
    
    # Run the simple app
    app = SimpleApp()
    app.run()

if __name__ == "__main__":
    main()