#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    file=~/.bash_profile
else
    file=~/.bashrc
fi

echo -e "export "PYTHONPATH=\"""\$PYTHONPATH:$PWD\""" >> $file
echo -e "now run this:\nsource $file"