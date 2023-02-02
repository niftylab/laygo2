
# Docs compilation

1. Make sure python-sphinx / sphinx_rtd_theme / myst_parser are installed.
   Otherwise, use the following commands to install all required packages and modules:
   ```bash
   $ yum install python-sphinx
   $ pip install sphinx_rtd_theme
   $ pip install pydata-sphinx-theme
   $ pip install myst_parser
   ```

2. Move to `laygo2/docs_workspace/docs_sphinx` and
   type the following cmd to compile:
   ```bash
   $ make clean html
   ```
