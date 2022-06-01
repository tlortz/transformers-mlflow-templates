from sys import version_info

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
        "pip",
        {"pip": ["mlflow",
                 "pandas",
                 "torch",
                 "transformers"]
        },
    ],
    "name": "sklearn_env"
}