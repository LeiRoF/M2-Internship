---
html_theme.sidebar_secondary.remove: true
html_theme.sidebar_primary.remove: true
---

# 👋 Getting Started

## 🔌 Installation

To run this project, you first need to install by yourself:

- [Python 3.10](https://www.python.org/downloads/)
- [Git CLI](https://git-scm.com/downloads)

```{admonition} Windows not supported
:class: warning

Due to the dependency to AMUSE which only work under Linux or MacOS, this project suffer the same limitation. If you are using Windows, you can use [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) to run this project.
```


Then, follow the steps below:

1. Install AMUSE dependencies by following the instructions on [this page](https://amuse.readthedocs.io/en/latest/install/howto-install-AMUSE.html)

    ````{admonition} Additional dependencies
    :class: warning

    On the system I used, the dependencies indicated by AMUSE documentation were not enough and I had to install the following dependencies. Unfortunately, there is no reliable method to install it properly... you will have to try and pray, good luck 🤞

    ```bash
    sudo apt-get install curl g++ gettext zlib1g-dev
    ```
    ````

2. Clone this repository:

    ```bash
    git clone https://github.com/Leirof/M2-Unveiling-3D-structure-of-pre-stellar-clouds
    ```

3. Move to the project directory:

    ```bash
    cd M2-Unveiling-3D-structure-of-pre-stellar-clouds
    ```

4. (optional but recommended) Create a virtual environment

    ```bash
    python -m venv venv
    ```

5. Activate the virtual environment

    ```bash
    source venv/bin/activate
    ```

6. Install the project dependencies:

    ```bash
    pip install -r requirements.txt
    ```




