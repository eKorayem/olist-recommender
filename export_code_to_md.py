import os
import json

ROOT_DIR = "."
OUTPUT_FILE = "project_code.md"

IGNORE_DIRS = {"__pycache__", ".git", ".vscode", ".claude", ".ipynb_checkpoints"}

with open(OUTPUT_FILE, "w", encoding="utf-8") as md_file:
    for root, dirs, files in os.walk(ROOT_DIR):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, ROOT_DIR)

            # Handle standard Python and .env files
            if file.endswith(".py") or file.startswith(".env"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    md_file.write(f"## {rel_path}\n\n")
                    md_file.write("```python\n")
                    md_file.write(content)
                    md_file.write("\n```\n\n")

                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")

            # Handle Jupyter Notebook files
            elif file.endswith(".ipynb"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        notebook = json.load(f)
                    
                    md_file.write(f"## {rel_path}\n\n")
                    md_file.write("```python\n")
                    
                    # Loop through cells and only extract source from code cells
                    for cell in notebook.get("cells", []):
                        if cell.get("cell_type") == "code":
                            # 'source' is a list of strings (lines of code)
                            source_code = "".join(cell.get("source", []))
                            if source_code.strip(): # Only write non-empty cells
                                md_file.write(source_code)
                                md_file.write("\n\n# --- End of Cell ---\n\n")
                                
                    md_file.write("\n```\n\n")

                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")

print(f"Done Exported all .py, .env, and .ipynb code to {OUTPUT_FILE}")