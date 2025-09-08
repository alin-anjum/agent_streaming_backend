import os
from pathlib import Path
from datetime import datetime

def concatenate_files(file_paths,
                      output_file="concatenated_files.txt",
                      include_metadata=True,
                      max_file_size_mb=10):
    """
    Concatenate multiple files into a single file with clear separators.

    Args
    ----
    file_paths : list[str | Path]
        File paths to concatenate.
    output_file : str
        Output file name.
    include_metadata : bool
        Whether to include file metadata (size, modified date).
    max_file_size_mb : int | float
        Maximum file size in MB to process (safety limit).
    """

    # ------------------------------------------------------------------ helpers
    def is_text_file(file_path):
        """Return True if extension / name looks like text source."""
        text_extensions = {
            '.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
            '.cu', '.md', '.rst', '.ini', '.cfg', '.conf', '.log', '.sql', '.sh',
            '.bat', '.cpp', '.c', '.h', '.java', '.php', '.rb', '.go', '.rs', '.ts',
            '.jsx', '.tsx', '.vue', '.scss', '.sass', '.less', '.env', '.gitignore',
            '.dockerfile'
        }
        file_path = Path(file_path)
        return (file_path.suffix.lower() in text_extensions or
                file_path.name.lower() in {'dockerfile', 'makefile',
                                           'readme', 'license', 'changelog'})

    def get_file_info(file_path):
        try:
            stat = file_path.stat()
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            return size, modified
        except Exception:
            return None, None

    def format_file_size(size_bytes):
        if size_bytes is None:
            return "0 B"
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 ** 2:
            return f"{size_bytes/1024:.1f} KB"
        if size_bytes < 1024 ** 3:
            return f"{size_bytes/(1024**2):.1f} MB"
        return f"{size_bytes/(1024**3):.1f} GB"

    # ----------------------------------------------------------------- counters
    total_files   = len(file_paths)
    processed     = 0
    skipped_count = 0
    error_count   = 0
    total_size    = 0

    error_paths   = []   #  <-- NEW
    skipped_paths = []   #  <-- NEW

    current_dir = Path.cwd()

    # ------------------------------------------------------------------ work
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # header ---------------------------------------------------------
        outfile.write("=" * 80 + "\n")
        outfile.write("CONCATENATED FILES\n")
        outfile.write("=" * 80 + "\n")
        outfile.write(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        outfile.write(f"Total files to process: {total_files}\n")
        outfile.write("=" * 80 + "\n\n")

        # loop ----------------------------------------------------------
        for idx, fp in enumerate(file_paths, 1):
            fp = Path(fp)

            # pretty path
            try:
                display_path = f"./{fp.relative_to(current_dir)}"
            except ValueError:
                display_path = str(fp.absolute())

            print(f"Processing {idx}/{total_files}: {display_path}")

            # separator header
            outfile.write("\n" + "╔" + "═" * 78 + "╗\n")
            outfile.write(f"║ FILE {idx:3d}/{total_files}: {display_path:<61} ║\n")

            # ------------------- existence
            if not fp.exists():
                outfile.write("║" + " " * 78 + "║\n")
                outfile.write("║ ❌ FILE NOT FOUND                                                    ║\n")
                outfile.write("╚" + "═" * 78 + "╝\n")
                error_count += 1
                error_paths.append(display_path)          #  <-- NEW
                continue

            # ------------------- metadata
            file_size, modified_date = get_file_info(fp)
            if file_size and file_size > max_file_size_mb * 1024 * 1024:
                outfile.write("║" + " " * 78 + "║\n")
                outfile.write(f"║ ⚠️  FILE TOO LARGE: {format_file_size(file_size):<50} ║\n")
                outfile.write("║ (Skipped - exceeds size limit)                                      ║\n")
                outfile.write("╚" + "═" * 78 + "╝\n")
                skipped_count += 1
                skipped_paths.append(display_path)         #  <-- NEW
                continue

            if include_metadata and file_size is not None:
                outfile.write("║" + " " * 78 + "║\n")
                outfile.write(f"║ Size: {format_file_size(file_size):<25} Modified: {modified_date:<20} ║\n")

            # ------------------- binary check
            if not is_text_file(fp):
                outfile.write("║" + " " * 78 + "║\n")
                outfile.write("║ ⚠️  BINARY FILE (Content not displayed)                             ║\n")
                outfile.write("╚" + "═" * 78 + "╝\n")
                skipped_count += 1
                skipped_paths.append(display_path)         #  <-- NEW
                continue

            outfile.write("╚" + "═" * 78 + "╝\n")

            # ------------------- read content
            try:
                content = None
                for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
                    try:
                        with open(fp, 'r', encoding=enc) as f_in:
                            content = f_in.read()
                        break
                    except UnicodeDecodeError:
                        continue

                if content is None:
                    outfile.write("❌ ERROR: Could not decode file with any encoding\n")
                    error_count += 1
                    error_paths.append(display_path)      #  <-- NEW
                    continue

                outfile.write(content)
                if content and not content.endswith('\n'):
                    outfile.write('\n')

                processed += 1
                if file_size:
                    total_size += file_size

            except Exception as exc:
                outfile.write(f"❌ ERROR: Could not read file - {exc}\n")
                error_count += 1
                error_paths.append(display_path)          #  <-- NEW

            # footer
            outfile.write("\n" + "╚" + "═" * 26 + f" END OF FILE {idx} " + "═" * 26 + "╝\n")

        # -------------------------------------------------------------- summary
        outfile.write("\n\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80 + "\n")
        outfile.write(f"Total files requested : {total_files}\n")
        outfile.write(f"Successfully processed: {processed}\n")
        outfile.write(f"Skipped (binary/large): {skipped_count}\n")
        outfile.write(f"Errors                : {error_count}\n")
        outfile.write(f"Total content size    : {format_file_size(total_size)}\n")
        outfile.write(f"Output file           : {output_file}\n")

        # list details
        if skipped_paths:
            outfile.write("\nSKIPPED FILES:\n")
            for p in skipped_paths:
                outfile.write(f"  - {p}\n")
        if error_paths:
            outfile.write("\nFILES WITH ERRORS:\n")
            for p in error_paths:
                outfile.write(f"  - {p}\n")

        outfile.write("=" * 80 + "\n")

    # --------------------------- console summary
    print("\n✅ Concatenation complete!")
    print(f"📄 Output saved to: {output_file}")
    print(f"📊 Processed: {processed}/{total_files} files")
    print(f"📦 Total size: {format_file_size(total_size)}")
    if skipped_count:
        print(f"⚠️  Skipped ({skipped_count}):")
        for p in skipped_paths:
            print(f"   • {p}")
    if error_count:
        print(f"❌ Errors ({error_count}):")
        for p in error_paths:
            print(f"   • {p}")

    # --------------------------- return dict
    return {
        'total': total_files,
        'processed': processed,
        'skipped': skipped_count,
        'skipped_paths': skipped_paths,   #  <-- NEW
        'errors': error_count,
        'error_paths': error_paths,       #  <-- NEW
        'total_size': total_size
    }

def concatenate_from_file_list(file_list_path, output_file="concatenated_files.txt", **kwargs):
    """
    Read file paths from a text file and concatenate them.
    
    Args:
        file_list_path: Path to text file containing list of file paths (one per line)
        output_file: Output file name
        **kwargs: Additional arguments for concatenate_files()
    """
    try:
        with open(file_list_path, 'r', encoding='utf-8') as f:
            file_paths = [
                line.strip() 
                for line in f.readlines() 
                if line.strip() and not line.strip().startswith('#')
            ]
        
        print(f"📋 Read {len(file_paths)} file paths from {file_list_path}")
        return concatenate_files(file_paths, output_file, **kwargs)
        
    except Exception as e:
        print(f"❌ Error reading file list: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example 1: Direct file list
    files_to_concatenate = [
        # "./data_utils/face_tracking/facemodel.py",
        # "./data_utils/face_tracking/util.py",
        # "./inference_system/inference_video.py",
        # "./inference_system/core/preprocessing_gpu.py",
        # "./inference_system/core/video_frame_manager.py",
        # "./inference_system/core/gpu_memory_manager.py",
        # "./inference_system/core/landmark_manager.py",
        # "./inference_system/utils/profiler.py",
        # "./data_utils/ave/audio.py",
        # "./data_utils/ave/models/audioEnc.py",
        "./train_328.py",
        "./unet_328.py",
        "./utils.py",
        "./datasetsss_328.py",
        "./syncnet_328.py",   
        "./training_system/training_utils.py",
        "./training_system/losses.py",
        "./training_system/training_config.py",
        "./training_system/multiscale_discriminator.py",
        "./training_system/traininglogger.py"             
    ]
    
    # Concatenate files
    result = concatenate_files(
        file_paths=files_to_concatenate,
        output_file="all_source_code.txt",
        include_metadata=True,
        max_file_size_mb=5  # Skip files larger than 5MB
    )
    
    