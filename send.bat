@echo off
SET /P variable="Did you remember to run tox?"

if %variable%==Y set variable=y
if %variable%==y (
    rm -r dist/*
    python setup.py bdist_wheel --universal
    python setup.py sdist
    twine upload dist/*
) else (
    echo upload cancelled
)