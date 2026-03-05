# This script captures the versions of the specified libraries and writes them to a requirements.txt file.
import pkg_resources

# Libraries used in your project
libraries = [
    'pandas',
    'numpy',
    'seaborn',
    'matplotlib',
    'scikit-learn',
    'shap',
    'shapiq',
    'joblib',
    'optuna',
    'glasspy',
    'tqdm',
    'scipy'
]

# Capture installed versions and write them to requirements.txt
with open('../requirements.txt', 'w') as f:
    for lib in libraries:
        try:
            version = pkg_resources.get_distribution(lib).version
            f.write(f"{lib}=={version}\n")
            print(f"{lib}=={version}")
        except pkg_resources.DistributionNotFound:
            print(f"{lib} not found")
            f.write(f"# {lib} not found in the environment\n")
        except Exception as e:
            print(f"Error with {lib}: {e}")
            f.write(f"# Error getting version for {lib}\n")

print("\nrequirements.txt file successfully created!")