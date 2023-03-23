import os

# Project information ---------------------------------------------------------

project = "Unveiling 3D structure of pre stellar cores"
copyright = "2023, Vincent Foriel"
author = "Vincent Foriel, Julien Montillaud"
html_favicon = "img/logo.png"

# -- General configuration ----------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinx_togglebutton',
    'sphinx_copybutton',
]
myst_heading_anchors = 6
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Options for HTML output -----------------------------------------------------

html_theme = 'pydata_sphinx_theme'

# html_css_files = [
#     'credits.css',
# ]

html_theme_options = {
    "github_url": "https://github.com/Leirof/M2-Unveiling-3D-structure-of-pre-stellar-clouds",
    "logo": {
        "image_dark": "_static/logo.png",
        "text": "",  # Uncomment to try text with logo
    }
}

html_logo = "img/logo.png"

# html_static_path = ['_static']

# html_css_files = [
#     'css/stylesheet.css',
# ]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]