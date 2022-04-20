#!/bin/sh

# Getting the needed inputs from the terminal
echo -n "Source File Path: "
read input_path

echo -n "Output File Path: "
read output_path

echo -n "Operation Mode: "
read mode

# Calling the Python interpreter and passing the arguments to the main program

if which python3 >/dev/null; then
  python3 main.py $input_path $output_path $mode
elif which python >/dev/null; then
  python main.py $input_path $output_path $mode
else
  echo "This machine don't have python"
fi
