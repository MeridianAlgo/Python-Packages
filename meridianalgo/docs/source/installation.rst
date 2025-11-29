.. _installation:

Installation
============

Prerequisites
------------

MeridianAlgo requires Python 3.7 or higher. We recommend using a virtual environment for installation.

Installing with pip
-------------------

The easiest way to install MeridianAlgo is using pip:

.. code-block:: bash

    pip install meridianalgo

This will install the core package along with all required dependencies.

Installing from source
----------------------

If you want to install the latest development version from source:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/MeridianAlgo/Python-Packages.git
       cd Python-Packages/meridianalgo

2. Install with pip in development mode:

   .. code-block:: bash

       pip install -e .

Optional Dependencies
--------------------

Some features require additional packages. You can install them with:

.. code-block:: bash

    # For machine learning features
    pip install meridianalgo[ml]
    
    # For optimization features
    pip install meridianalgo[optimization]
    
    # For all optional dependencies
    pip install meridianalgo[all]

Verifying the Installation
-------------------------

To verify that MeridianAlgo is installed correctly, run the following in a Python interpreter:

.. code-block:: python

    import meridianalgo as ma
    print(f"MeridianAlgo version: {ma.__version__}")

Troubleshooting
--------------

If you encounter any issues during installation, please check the following:

- Ensure you have the latest version of pip:
  
  .. code-block:: bash

      pip install --upgrade pip

- If you get permission errors, try installing with the `--user` flag:
  
  .. code-block:: bash

      pip install --user meridianalgo

- For issues with specific dependencies, try installing them manually first.

Getting Help
-----------

If you need help with the installation, please open an issue on our `GitHub repository <https://github.com/MeridianAlgo/Python-Packages/issues>`_.
