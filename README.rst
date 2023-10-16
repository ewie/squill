======
squill
======

Squill is a tool for managing SQL database revisions.


Development Setup
=================

Create and activate the virtual environment:

.. code-block:: sh

   python3 -m venv venv
   . venv/bin/activate

Install development dependencies and squill in editable mode:

.. code-block:: sh

   pip install -U pip
   pip install -r requirements/dev.txt
   pip install -e .

Build and publish:

.. code-block:: sh

   python -m build
   twine upload dist/*
