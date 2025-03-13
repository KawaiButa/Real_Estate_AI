# seed.py
import glob
import importlib
import os
import pathlib

def main() -> None:
    # Get absolute path to the seeders folder
    seeders_dir = pathlib.Path(__file__).parent / "seeders"
    
    # Find all .py files inside the seeders directory (except __init__.py)
    pattern = str(seeders_dir / "*.py")
    files = glob.glob(pattern)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        if filename == "__init__.py":
            continue  # skip __init__.py

        # Convert "roles.py" -> "roles" (the module name)
        module_name = filename.replace(".py", "")

        # Import using the full dotted path e.g. "my_ seeders.roles"
        # Adjust "my_app" below to match your actual import path
        import_path = f"database.seeders.{module_name}"
        module = importlib.import_module(import_path)

        # If there's a function named 'run' (or 'seed'), call it
        if hasattr(module, "run"):
            print(f"Running seeder: {module_name}")
            module.run()
        else:
            print(f"Skipping {module_name} (no run() function).")

if __name__ == "__main__":
    main()
