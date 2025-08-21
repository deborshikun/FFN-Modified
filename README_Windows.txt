---------------------------------------------------------------------------------

FRESH START - Remove existing venv and create new one:

1. Deactivate current venv (if active):
   deactivate

2. Remove the venv folder:
  rmdir -r venv

3. Create new virtual environment:

Make virtual environment using Python 3.10.11

py -3.10 -m venv venv
----------------------------------------

Temporarily allow script execution for this session
Run this command in your PowerShell before activating your venv:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

Then activate your venv:

venv\Scripts\activate

python -m pip install --upgrade pip

----------------------------------------

Install the required packages


pip install numpy==1.24.3 onnx==1.14.1 onnxruntime==1.15.1 torch torchvision scikit-learn matplotlib graphviz

Also install graphviz on pc using https://graphviz.org/download/
-----------------------------------------

Verify your environment


python --version
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import onnx; print('ONNX:', onnx.__version__)"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"


You should see Python 3.10.x and numpy 1.24.3.

---------------------------------------

python loop_single_instance.py

----------------------------------------------------------------------------------

- Always activate your venv before running your scripts.

venv\Scripts\activate

- Use `python`, not `py`, after activation to ensure you use the venv’s interpreter.

