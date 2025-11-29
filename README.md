# Prerequisites

## Python Installation

1. Install python > 3.13 using pyenv: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation
2. Enable python: `pyenv local 3.13.7`

## Package Manager Installation

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/#installation-methods

## Ollama Installation

1. Install Ollama on MAC: https://ollama.com/download
2. Download the open source model that is supported according to your architecture. For eg: llama3.1

# Build and Run

## Sync dependencies

1. Run the command: `uv sync`

## Run the project in the virtual environment

1. Rename `config.template.toml` to `config.toml` and fill its contents.
2. Run the command: `uv run main.py`. It will sync the dependencies, enable the virtual environment and run the command in that virtual environment.
