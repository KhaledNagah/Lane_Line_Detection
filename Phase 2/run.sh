#!/bin/sh

# Getting the needed inputs from the terminal
echo -n "Source File Path: "
read input_path

echo -n "Output File Path: "
read output_path

# Calling the Python interpreter and passing the arguments to the main program
python3 main.py $input_path $output_path
