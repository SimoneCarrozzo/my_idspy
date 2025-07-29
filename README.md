# IDS Framework

A comprehensive framework for building and evaluating Intrusion Detection Systems (IDS).

## Overview

This framework provides a structured approach to developing intrusion detection solutions, with core functionality organized in the `src` folder. It includes tools for network data preprocessing, attack detection modeling, evaluation metrics, and result visualization tailored specifically for IDS workflows.

## Project Structure

```
ids-framework/
├── src/                    # Core framework functions and modules
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Simone-Albero/idspy.git
   cd idspy
   ```

2. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can import and use the core functions from the `src` folder:

```python
# Example usage (adjust imports based on your actual src structure)
from src import your_module
```