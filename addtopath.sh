#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    file=~/.bash_profile
else
    file=~/.bashrc
fi

str="export "PYTHONPATH=\"""\$PYTHONPATH:$PWD\"""

if ! grep -Fxq "$str" $file 
then
    echo -e "$str" >> $file
    echo -e "now run this:\nsource $file"
fi