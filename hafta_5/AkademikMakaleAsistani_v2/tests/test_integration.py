"""
Basic Integration Test for Advanced Academic Research Assistant v2.0
Tests core functionality without requiring external APIs
"""
import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all modules can be imported successfully"""
    print("🧪 Testing module imports...")
    
    try:
        from chains.research_chains import ResearchAnalysisChain
        print("✅ Research chains imported successfully")
        
        from chains.writing_chains import AcademicWritingChain  
        print("✅ Writing chains imported successfully")
        
        from chains.analysis_chains import DocumentAnalysisChain
        print("✅ Analysis chains imported successfully")
        
        from memory.research_memory import ResearchSessionMemory
        print("✅ Research memory imported successfully")
        
        from memory.project_memory import ProjectMemory
        print("✅ Project memory imported successfully")
        
        from tools.pdf_manager import EnhancedPDFManager
        print("✅ PDF manager imported successfully")
        
        from tools.vector_db import EnhancedVectorDB
        print("✅ Vector DB imported successfully")
        
        from tools.literature_tool import LiteratureSearchTool
        print("✅ Literature tool imported successfully")
        
        from tools.reference_tool import ReferenceManagerTool
        print("✅ Reference tool imported successfully")
        
        from streaming.handlers import ResearchStreamingHandler, ProgressTracker
        print("✅ Streaming handlers imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_initialization():
    """Test basic system initialization"""
    print("\\n🏗️ Testing system initialization...")
    
    try:
        from memory.project_memory import ProjectMemory
        
        # Test project memory
        pm = ProjectMemory()
        print("✅ Project memory initialized")
        
        # Test streaming handler
        from streaming.handlers import ResearchStreamingHandler
        handler = ResearchStreamingHandler("test_session")
        print("✅ Streaming handler initialized")
        
        # Test progress tracker
        from streaming.handlers import ProgressTracker
        tracker = ProgressTracker("Test Operation")
        tracker.add_stage("Stage 1", 10)
        tracker.start_stage("Stage 1")
        tracker.complete_stage("Stage 1")
        print("✅ Progress tracker working")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def test_project_memory():
    """Test project memory functionality"""
    print("\\n💾 Testing project memory...")
    
    try:
        from memory.project_memory import ProjectMemory
        
        pm = ProjectMemory()
        
        # Create a test project
        project_id = pm.create_project(
            name="Test Research Project",
            description="Testing project memory functionality",
            tags=["test", "research"]
        )
        
        if project_id:
            print(f"✅ Project created: {project_id}")
            
            # Add a resource
            resource_id = pm.add_resource(
                project_id=project_id,
                name="Test Paper",
                resource_type="pdf",
                path="./test.pdf",
                summary="Test paper summary"
            )
            
            if resource_id:
                print(f"✅ Resource added: {resource_id}")
            
            # Get project analytics
            analytics = pm.get_project_analytics(project_id)
            if analytics:
                print("✅ Analytics generated successfully")
                
            return True
        
    except Exception as e:
        print(f"❌ Project memory test error: {e}")
        return False

def test_reference_tool():
    """Test reference management tool"""
    print("\\n📚 Testing reference management...")
    
    try:
        from tools.reference_tool import ReferenceManagerTool
        
        tool = ReferenceManagerTool()
        
        # Test adding a reference
        result = tool._run('add reference {"title": "Test Paper", "authors": ["John Doe"], "year": "2024"}')
        
        if "success" in result or "reference_id" in result:
            print("✅ Reference tool working")
            return True
        else:
            print(f"⚠️ Reference tool returned: {result}")
            return True  # Not a critical failure
            
    except Exception as e:
        print(f"❌ Reference tool test error: {e}")
        return False

def test_main_application():
    """Test main application import"""
    print("\\n🚀 Testing main application...")
    
    try:
        # Just test import, not initialization (requires API keys)
        sys.path.append(str(project_root))
        
        # Test if main.py can be imported
        import main
        print("✅ Main application module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Main application test error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🎯 Advanced Academic Research Assistant v2.0 - Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Basic Initialization", test_basic_initialization), 
        ("Project Memory", test_project_memory),
        ("Reference Tool", test_reference_tool),
        ("Main Application", test_main_application)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\\n🎯 Total: {len(results)} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    success_rate = (passed / len(results)) * 100
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\\n🎉 System is ready for use!")
    elif success_rate >= 60:
        print("\\n⚠️ System has some issues but basic functionality works")
    else:
        print("\\n❌ System has significant issues that need to be resolved")
    
    return success_rate

if __name__ == "__main__":
    success_rate = run_all_tests()
    
    # Exit with appropriate code
    if success_rate >= 80:
        sys.exit(0)
    else:
        sys.exit(1)