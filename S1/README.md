## Prerequisites

### Python Dependencies
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

### System Dependencies
This project requires TeX Live for generating publication-quality plots. Install it using:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

#### macOS
```bash
brew install --cask mactex
```

#### Windows
Download and install TeX Live from: https://tug.org/texlive/windows.html

## Use of AI Tools

The following AI tools were used in the development of this project:

- **Cursor (Model: Claude 3.5 Sonnet)**: 
  - Provided code completion suggestions.
  - Provided debugging assistance.
  - Helped with implementation of small ideas across the project.
- **ChatGPT-4.0**: 
  - Provided explanations on development questions.
  - Generated some docstrings.
  - Helped optimize the `.pyx` file to use type declarations and 
  - Helped with "mystyle.mplstyle" file.
  - Helped with "README.md" file.