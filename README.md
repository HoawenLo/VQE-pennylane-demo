# Variational quantum eigensolver demonstration with Pennylane

Variational quantum eigensolver is a hybrid classical-quantum algoritm which aims to find the ground state energy level of a quantum system.

Like other variational quantum algorithms, VQE utilises optimisation concecpts from neural networks, to apply to parameterised quantum circuits.

## Getting started

1) Clone the repository.

Use the following git command on command line
```
git clone https://github.com/HoawenLo/VQE-pennylane-demo.git
```

2) Setting up a virtual environment and pip installing the required python packages.

Check python 3.10 is installed. In your command line environment type
```
python
```

or

```
python3
```

This should give you the following
```
Python 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

After verifying that type
```
>>> quit()
```

Navigate to the highest level of the project directory.
```
cd path/to/highest/level/project
```

This however should just be
```
cd VQE-pennylane-demo
```
as you have just cloned the repository.

Create the python virtual environment and activate it.
```
python -m venv name_of_virtual_environment
```

If on linux
```
source name_of_virtual_environment/bin/activate
```

On windows
```
name_of_virtual_environment\Scripts\activate
```

Add virtual environment to .gitignore. Use one of the following

```
code .gitignore # Opens Vscode editor
kate .gitignore # Linux
notepad .gitignore # Windows
```

Add name of virtual environment
```
...
*.pyc
name_of_virtual_environment # Add here
...

Pip install the requirements
```
pip3 install -r requirements.txt
```

With that you should be ready.