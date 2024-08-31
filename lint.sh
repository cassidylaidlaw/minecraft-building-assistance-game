#!/bin/bash

black mbag stubs tests
isort mbag stubs tests
flake8 mbag stubs tests
MYPYPATH=`pwd`/stubs mypy mbag tests
