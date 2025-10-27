"""
Verify project structure and fix common issues
Run from project root: python verify_project_structure.py
"""

import os
from pathlib import Path
import sys

def check_directory_structure():
    """Check if all required directories exist"""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60 + "\n")
    
    required_dirs = [
        "backend",
        "backend/models",
        "backend/ingestion",
        "backend/search",
        "backend/api",
        "data",
        "data/videos",
        "data/embeddings",
        "data/frames",
        "models",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"❌ {dir_path} (MISSING)")
            all_exist = False
            # Create missing directory
            path.mkdir(parents=True, exist_ok=True)
            print(f"  → Created {dir_path}")
    
    return all_exist

def check_required_files():
    """Check if all required files exist"""
    print("\n" + "="*60)
    print("CHECKING REQUIRED FILES")
    print("="*60 + "\n")
    
    required_files = {
        ".env": "Configuration file",
        "backend/models/model_loader.py": "Model loader module",
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {file_path} ({size} bytes) - {description}")
        else:
            print(f"❌ {file_path} (MISSING) - {description}")
            all_exist = False
    
    return all_exist

def fix_init_files():
    """Create properly encoded __init__.py files"""
    print("\n" + "="*60)
    print("FIXING __init__.py FILES")
    print("="*60 + "\n")
    
    init_files = [
        "backend/__init__.py",
        "backend/models/__init__.py",
        "backend/ingestion/__init__.py",
        "backend/search/__init__.py",
        "backend/api/__init__.py",
    ]
    
    for file_path in init_files:
        path = Path(file_path)
        try:
            # Create or overwrite with empty content (UTF-8 encoded)
            with open(path, 'w', encoding='utf-8') as f:
                f.write("")  # Empty file
            print(f"✓ Fixed {file_path}")
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")

def test_imports():
    """Test if Python can import the modules"""
    print("\n" + "="*60)
    print("TESTING MODULE IMPORTS")
    print("="*60 + "\n")
    
    # Add backend to path
    sys.path.insert(0, 'backend')
    
    tests_passed = True
    
    # Test 1: Import backend
    try:
        import backend
        print("✓ backend module")
    except Exception as e:
        print(f"❌ backend module: {e}")
        tests_passed = False
    
    # Test 2: Import model_loader
    try:
        from models.model_loader import get_model_manager
        print("✓ models.model_loader module")
    except Exception as e:
        print(f"❌ models.model_loader module: {e}")
        tests_passed = False
        import traceback
        traceback.print_exc()
    
    return tests_passed

def check_environment():
    """Check environment configuration"""
    print("\n" + "="*60)
    print("CHECKING ENVIRONMENT")
    print("="*60 + "\n")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        "YOLO_MODEL",
        "CLIP_MODEL",
        "DEVICE",
        "VIDEO_PATH",
        "FRAMES_PATH",
        "EMBEDDINGS_PATH",
    ]
    
    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var} = {value}")
        else:
            print(f"❌ {var} (NOT SET)")
            all_set = False
    
    return all_set

def create_sample_test():
    """Create a minimal test file"""
    print("\n" + "="*60)
    print("CREATING MINIMAL TEST FILE")
    print("="*60 + "\n")
    
    test_content = '''"""
Minimal test to verify basic functionality
"""

print("Testing imports...")

try:
    import sys
    sys.path.insert(0, 'backend')
    
    from models.model_loader import ModelManager
    print("✓ ModelManager imported successfully")
    
    manager = ModelManager()
    print("✓ ModelManager instance created")
    
    print("\\n✅ Basic imports working!")
    print("\\nNext: Run 'python test_models.py' to test model loading")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open("test_minimal.py", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print("✓ Created test_minimal.py")
    print("  Run with: python test_minimal.py")

def main():
    print("\n" + "="*60)
    print("PROJECT STRUCTURE VERIFICATION")
    print("="*60)
    
    # Step 1: Check directories
    dirs_ok = check_directory_structure()
    
    # Step 2: Check files
    files_ok = check_required_files()
    
    # Step 3: Fix __init__.py files
    fix_init_files()
    
    # Step 4: Check environment
    env_ok = check_environment()
    
    # Step 5: Test imports
    imports_ok = test_imports()
    
    # Step 6: Create minimal test
    create_sample_test()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60 + "\n")
    
    if all([dirs_ok, files_ok, env_ok, imports_ok]):
        print("✅ PROJECT STRUCTURE IS CORRECT!")
        print("\nNext steps:")
        print("  1. Run: python test_minimal.py")
        print("  2. Then run: python test_models.py")
        print("  3. Add a video to data/videos/")
    else:
        print("⚠️  SOME ISSUES FOUND:")
        if not dirs_ok:
            print("  - Directory structure incomplete (fixed automatically)")
        if not files_ok:
            print("  - Required files missing")
        if not env_ok:
            print("  - Environment configuration issues")
        if not imports_ok:
            print("  - Module import errors")
        
        print("\nPlease check the errors above and try again.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()