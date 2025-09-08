import os
from pathlib import Path
import glob

# Folders and files to ignore
IGNORE_PATTERNS = {
    '__pycache__',
    '.git',
    '.gitignore',
    '.vscode',
    '.idea',
    '.pytest_cache',
    '.mypy_cache',
    '.tox',
    '.coverage',
    'node_modules',
    '.DS_Store',
    'Thumbs.db',
    '.env',
    '.venv',
    'venv',
    'env',
    '__MACOSX',
    '.cache',
    '.tmp',
    'tmp',
    '.log'
}

def should_ignore(path):
    """Check if a path should be ignored"""
    return path.name in IGNORE_PATTERNS or path.name.startswith('.')

def expand_patterns(patterns):
    """Expand glob patterns into actual file paths"""
    expanded_files = []
    for pattern in patterns:
        # Use pathlib for glob matching
        matches = list(Path('.').glob(pattern))
        for match in matches:
            if not should_ignore(match) and match.is_file():
                expanded_files.append(match)
    return expanded_files

def get_relative_path_display(folder_path):
    """Get a clean relative path for display"""
    current_dir = Path.cwd()
    try:
        rel_path = folder_path.resolve().relative_to(current_dir)
        return f"./{rel_path}"
    except ValueError:
        return str(folder_path.resolve())

def collect_all_paths(path, max_depth=None, depth=0, current_folder_base=""):
    """Recursively collect all file and folder paths for simple output"""
    if max_depth is not None and depth > max_depth:
        return []
    
    all_paths = []
    path = Path(path)
    
    if not path.exists():
        return []
    
    if path.is_file():
        return [f"{current_folder_base}/{path.name}".lstrip("/")]
    
    try:
        items = [item for item in path.iterdir() if not should_ignore(item)]
        items.sort(key=lambda x: (x.is_file(), x.name.lower()))
        
        for item in items:
            if current_folder_base:
                item_path = f"{current_folder_base}/{item.name}"
            else:
                item_path = item.name
            
            if item.is_dir():
                all_paths.append(f"{item_path}/")
                # Recursively collect from subdirectories
                sub_paths = collect_all_paths(item, max_depth, depth + 1, item_path)
                all_paths.extend(sub_paths)
            else:
                all_paths.append(item_path)
                
    except PermissionError:
        pass
        
    return all_paths

def collect_filtered_paths(path, max_depth=None, depth=0, current_folder_base="", max_other_files=10):
    """Recursively collect filtered paths for simple output"""
    if max_depth is not None and depth > max_depth:
        return []
    
    all_paths = []
    path = Path(path)
    
    if not path.exists():
        return []
    
    if path.is_file():
        return [f"{current_folder_base}/{path.name}".lstrip("/")]
    
    try:
        items = [item for item in path.iterdir() if not should_ignore(item)]
        
        # Separate directories, .py files, and other files
        directories = [item for item in items if item.is_dir()]
        py_files = [item for item in items if item.is_file() and item.suffix == '.py']
        other_files = [item for item in items if item.is_file() and item.suffix != '.py']
        
        # Sort each category
        directories.sort(key=lambda x: x.name.lower())
        py_files.sort(key=lambda x: x.name.lower())
        other_files.sort(key=lambda x: x.name.lower())
        
        # Limit other files
        other_files_display = other_files[:max_other_files]
        other_files_hidden = len(other_files) - len(other_files_display)
        
        # Process directories
        for item in directories:
            if current_folder_base:
                item_path = f"{current_folder_base}/{item.name}"
            else:
                item_path = item.name
            
            all_paths.append(f"{item_path}/")
            # Recursively collect from subdirectories
            sub_paths = collect_filtered_paths(item, max_depth, depth + 1, item_path, max_other_files)
            all_paths.extend(sub_paths)
        
        # Process Python files
        for item in py_files:
            if current_folder_base:
                item_path = f"{current_folder_base}/{item.name}"
            else:
                item_path = item.name
            all_paths.append(f"{item_path} [PYTHON]")
        
        # Process other files
        for item in other_files_display:
            if current_folder_base:
                item_path = f"{current_folder_base}/{item.name}"
            else:
                item_path = item.name
            all_paths.append(item_path)
        
        # Add note about hidden files
        if other_files_hidden > 0:
            all_paths.append(f"[{other_files_hidden} more non-Python files hidden]")
                
    except PermissionError:
        pass
        
    return all_paths

def create_tree_structure(folders, file_patterns=None, output_file="folder_hierarchy.txt", simple_output_file="folder_hierarchy_simple.txt", max_depth=None):
    """Create both visual and simple hierarchy"""
    
    def get_tree_structure(path, prefix="", depth=0):
        """Recursively build tree structure"""
        if max_depth is not None and depth > max_depth:
            return []
        
        lines = []
        path = Path(path)
        
        if not path.exists():
            return [f"{prefix}[PATH NOT FOUND: {path}]"]
        
        if path.is_file():
            return [f"{prefix}{path.name}"]
        
        try:
            items = [item for item in path.iterdir() if not should_ignore(item)]
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                
                item_name = f"{item.name}/" if item.is_dir() else item.name
                lines.append(f"{prefix}{current_prefix}{item_name}")
                
                if item.is_dir():
                    sub_lines = get_tree_structure(item, prefix + next_prefix, depth + 1)
                    lines.extend(sub_lines)
                    
        except PermissionError:
            lines.append(f"{prefix}[PERMISSION DENIED]")
            
        return lines
    
    # Generate visual tree
    visual_lines = []
    simple_lines = []
    
    # Process folders
    for folder in folders:
        folder_path = Path(folder)
        display_path = get_relative_path_display(folder_path)
        
        # Visual format
        visual_lines.append(f"\n📁 {display_path}")
        visual_lines.append("=" * 50)
        
        # Simple format
        simple_lines.append(f"\nFolder: {display_path}")
        simple_lines.append("-" * 30)
        
        if folder_path.exists():
            # Visual tree
            tree_lines = get_tree_structure(folder_path)
            if tree_lines:
                visual_lines.extend(tree_lines)
            else:
                visual_lines.append("[EMPTY DIRECTORY]")
            
            # Simple paths
            simple_paths = collect_all_paths(folder_path, max_depth)
            if simple_paths:
                simple_lines.extend(simple_paths)
            else:
                simple_lines.append("[EMPTY DIRECTORY]")
        else:
            visual_lines.append("[FOLDER NOT FOUND]")
            simple_lines.append("[FOLDER NOT FOUND]")
        
        visual_lines.append("")
        simple_lines.append("")
    
    # Process file patterns
    if file_patterns:
        pattern_files = expand_patterns(file_patterns)
        if pattern_files:
            visual_lines.append(f"\n📄 Pattern-matched files")
            visual_lines.append("=" * 50)
            simple_lines.append(f"\nPattern-matched files:")
            simple_lines.append("-" * 30)
            
            # Group files by pattern for better organization
            for pattern in file_patterns:
                pattern_matches = [f for f in pattern_files if f.match(pattern)]
                if pattern_matches:
                    visual_lines.append(f"\nPattern: {pattern}")
                    simple_lines.append(f"\nPattern: {pattern}")
                    
                    pattern_matches.sort(key=lambda x: x.name.lower())
                    for i, file_path in enumerate(pattern_matches):
                        is_last = i == len(pattern_matches) - 1
                        prefix = "└── " if is_last else "├── "
                        
                        # Add special marker for Python files
                        if file_path.suffix == '.py':
                            file_display = f"🐍 {file_path.name}"
                        else:
                            file_display = file_path.name
                            
                        visual_lines.append(f"{prefix}{file_display}")
                        simple_lines.append(f"{file_path.name}")
            
            visual_lines.append("")
            simple_lines.append("")
    
    # Write visual file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(visual_lines))
    
    # Write simple file
    with open(simple_output_file, 'w', encoding='utf-8') as f:
        f.write("# Simple File Structure\n")
        f.write("# Same content as visual hierarchy but in plain text\n")
        f.write("# " + "="*60 + "\n")
        f.write("\n".join(simple_lines))
    
    print(f"✅ Visual hierarchy saved to: {output_file}")
    print(f"✅ Simple hierarchy saved to: {simple_output_file}")

def create_filtered_tree_structure(folders, file_patterns=None, output_file="folder_hierarchy.txt", simple_output_file="folder_hierarchy_simple.txt", max_depth=None, max_other_files=10):
    """Create both visual and simple filtered hierarchy"""
    
    def get_filtered_tree_structure(path, prefix="", depth=0):
        """Visual filtered tree structure"""
        if max_depth is not None and depth > max_depth:
            return []
        
        lines = []
        path = Path(path)
        
        if not path.exists():
            return [f"{prefix}[PATH NOT FOUND: {path}]"]
        
        if path.is_file():
            return [f"{prefix}{path.name}"]
        
        try:
            items = [item for item in path.iterdir() if not should_ignore(item)]
            
            directories = [item for item in items if item.is_dir()]
            py_files = [item for item in items if item.is_file() and item.suffix == '.py']
            other_files = [item for item in items if item.is_file() and item.suffix != '.py']
            
            directories.sort(key=lambda x: x.name.lower())
            py_files.sort(key=lambda x: x.name.lower())
            other_files.sort(key=lambda x: x.name.lower())
            
            other_files_display = other_files[:max_other_files]
            other_files_hidden = len(other_files) - len(other_files_display)
            
            items_to_display = directories + py_files + other_files_display
            
            for i, item in enumerate(items_to_display):
                is_last_item = (i == len(items_to_display) - 1) and (other_files_hidden == 0)
                current_prefix = "└── " if is_last_item else "├── "
                next_prefix = "    " if is_last_item else "│   "
                
                if item.is_dir():
                    item_name = f"{item.name}/"
                    lines.append(f"{prefix}{current_prefix}{item_name}")
                    sub_lines = get_filtered_tree_structure(item, prefix + next_prefix, depth + 1)
                    lines.extend(sub_lines)
                else:
                    if item.suffix == '.py':
                        item_name = f"🐍 {item.name}"
                    else:
                        item_name = item.name
                    lines.append(f"{prefix}{current_prefix}{item_name}")
            
            if other_files_hidden > 0:
                hidden_prefix = "└── " if len(items_to_display) == 0 else "└── "
                lines.append(f"{prefix}{hidden_prefix}... and {other_files_hidden} more non-Python files")
                    
        except PermissionError:
            lines.append(f"{prefix}[PERMISSION DENIED]")
            
        return lines
    
    # Read existing content
    visual_content = ""
    simple_content = ""
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            visual_content = f.read()
    
    if os.path.exists(simple_output_file):
        with open(simple_output_file, 'r', encoding='utf-8') as f:
            simple_content = f.read()
    
    # Generate filtered trees
    visual_lines = []
    simple_lines = []
    
    visual_lines.append("\n" + "=" * 80)
    visual_lines.append("🐍 PYTHON FILES + LIMITED OTHER FILES VIEW")
    visual_lines.append("=" * 80)
    
    simple_lines.append("\n" + "=" * 60)
    simple_lines.append("PYTHON FILES + LIMITED OTHER FILES VIEW")
    simple_lines.append("=" * 60)
    
    # Process folders
    for folder in folders:
        folder_path = Path(folder)
        display_path = get_relative_path_display(folder_path)
        
        # Visual format
        visual_lines.append(f"\n📁 {display_path} (Python + {max_other_files} other files per folder)")
        visual_lines.append("-" * 50)
        
        # Simple format
        simple_lines.append(f"\nFolder: {display_path} (Python + {max_other_files} other files per folder)")
        simple_lines.append("-" * 40)
        
        if folder_path.exists():
            # Visual tree
            tree_lines = get_filtered_tree_structure(folder_path)
            if tree_lines:
                visual_lines.extend(tree_lines)
            else:
                visual_lines.append("[EMPTY DIRECTORY]")
            
            # Simple paths
            simple_paths = collect_filtered_paths(folder_path, max_depth, max_other_files=max_other_files)
            if simple_paths:
                simple_lines.extend(simple_paths)
            else:
                simple_lines.append("[EMPTY DIRECTORY]")
        else:
            visual_lines.append("[FOLDER NOT FOUND]")
            simple_lines.append("[FOLDER NOT FOUND]")
        
        visual_lines.append("")
        simple_lines.append("")
    
    # Process file patterns for filtered view
    if file_patterns:
        pattern_files = expand_patterns(file_patterns)
        if pattern_files:
            visual_lines.append(f"\n📄 Pattern-matched files (Filtered View)")
            visual_lines.append("-" * 50)
            simple_lines.append(f"\nPattern-matched files (Filtered View):")
            simple_lines.append("-" * 40)
            
            for pattern in file_patterns:
                pattern_matches = [f for f in pattern_files if f.match(pattern)]
                if pattern_matches:
                    # Only show Python files and limited others in filtered view
                    py_matches = [f for f in pattern_matches if f.suffix == '.py']
                    other_matches = [f for f in pattern_matches if f.suffix != '.py'][:max_other_files]
                    
                    visual_lines.append(f"\nPattern: {pattern}")
                    simple_lines.append(f"\nPattern: {pattern}")
                    
                    all_matches = py_matches + other_matches
                    all_matches.sort(key=lambda x: x.name.lower())
                    
                    for i, file_path in enumerate(all_matches):
                        is_last = i == len(all_matches) - 1
                        prefix = "└── " if is_last else "├── "
                        
                        if file_path.suffix == '.py':
                            file_display = f"🐍 {file_path.name}"
                            simple_display = f"{file_path.name} [PYTHON]"
                        else:
                            file_display = file_path.name
                            simple_display = file_path.name
                            
                        visual_lines.append(f"{prefix}{file_display}")
                        simple_lines.append(simple_display)
                    
                    # Show count of hidden files
                    hidden_count = len(pattern_matches) - len(all_matches)
                    if hidden_count > 0:
                        visual_lines.append(f"└── ... and {hidden_count} more files")
                        simple_lines.append(f"[{hidden_count} more files hidden]")
            
            visual_lines.append("")
            simple_lines.append("")
    
    # Append to files
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(visual_content)
        f.write("\n".join(visual_lines))
    
    with open(simple_output_file, 'w', encoding='utf-8') as f:
        f.write(simple_content)
        f.write("\n".join(simple_lines))
    
    print(f"✅ Filtered visual hierarchy appended to: {output_file}")
    print(f"✅ Filtered simple hierarchy appended to: {simple_output_file}")

# Example usage
if __name__ == "__main__":
    # Specify the folders for full hierarchy
    folders_full_hierarchy = [
        "./data_utils",        
        "./inference_system",
    ]
    
    # Specify file patterns to include
    file_patterns = [
        "*.py",           # All Python files in root
        "*.txt",          # All text files in root
        "*.md",           # All markdown files in root
        "*.yaml",         # All YAML files in root
        "*.yml",          # All YML files in root
        "*.json",         # All JSON files in root
    ]
    
    # Specify the folders for filtered view
    folders_filtered_view = [
       "./dataset",                
        # "./data",
        # "./output",
        # "./submodules",
        # "./audios",
        # "./gridencoder",
    ]
    
    print("🚀 Generating both visual and simple file structures...")
    
    # Create both visual and simple versions with file patterns
    create_tree_structure(
        folders=folders_full_hierarchy,
        file_patterns=file_patterns,  # Now includes pattern matching!
        output_file="project_structure.txt",
        simple_output_file="project_structure_simple.txt",
        max_depth=3
    )
    
    create_filtered_tree_structure(
        folders=folders_filtered_view,
        file_patterns=file_patterns,  # Pattern matching in filtered view too!
        output_file="project_structure.txt",
        simple_output_file="project_structure_simple.txt",
        max_depth=3,
        max_other_files=10
    )
    
    print("\n🎉 Both file structures created!")
    print("📄 Files generated:")
    print("  - project_structure.txt (visual tree)")
    print("  - project_structure_simple.txt (simple paths)")