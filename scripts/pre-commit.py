#!/usr/bin/env python3
"""
Pre-commit script to run clang-format and clang-tidy on staged files.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def get_staged_cpp_files():
    """Get list of staged C++ files."""
    success, stdout, _ = run_command("git diff --cached --name-only --diff-filter=ACM")
    if not success:
        return []
    
    files = []
    for line in stdout.strip().split('\n'):
        if line and any(line.endswith(ext) for ext in ['.cpp', '.hpp', '.h', '.c', '.cc', '.cxx']):
            if os.path.exists(line):
                files.append(line)
    
    return files

def run_clang_format(files):
    """Run clang-format on the given files."""
    if not files:
        return True
    
    print("Running clang-format...")
    for file in files:
        success, _, stderr = run_command(f"clang-format -i {file}")
        if not success:
            print(f"clang-format failed for {file}: {stderr}")
            return False
        
        # Stage the formatted file
        run_command(f"git add {file}")
    
    print("✓ clang-format completed successfully")
    return True

def run_clang_tidy(files):
    """Run clang-tidy on the given files."""
    if not files:
        return True
    
    # Check if compile_commands.json exists
    if not os.path.exists("build/compile_commands.json"):
        print("Warning: build/compile_commands.json not found. Run 'cmake -B build' first.")
        print("Skipping clang-tidy check.")
        return True
    
    print("Running clang-tidy...")
    source_files = [f for f in files if f.endswith(('.cpp', '.c', '.cc', '.cxx'))]
    
    for file in source_files:
        success, _, stderr = run_command(f"clang-tidy -p build {file}")
        if not success:
            print(f"clang-tidy failed for {file}:")
            print(stderr)
            return False
    
    print("✓ clang-tidy completed successfully")
    return True

def main():
    """Main pre-commit function."""
    print("Running pre-commit checks...")
    
    staged_files = get_staged_cpp_files()
    if not staged_files:
        print("No C++ files staged for commit.")
        return 0
    
    print(f"Checking {len(staged_files)} file(s):")
    for file in staged_files:
        print(f"  - {file}")
    
    # Run clang-format
    if not run_clang_format(staged_files):
        print("❌ Pre-commit checks failed: clang-format errors")
        return 1
    
    # Run clang-tidy
    if not run_clang_tidy(staged_files):
        print("❌ Pre-commit checks failed: clang-tidy errors")
        print("\nTo fix clang-tidy issues:")
        print("  1. Review the errors above")
        print("  2. Fix the code or add NOLINT comments where appropriate")
        print("  3. Re-run the commit")
        return 1
    
    print("✅ All pre-commit checks passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
