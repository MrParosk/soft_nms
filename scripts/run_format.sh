#!/bin/bash

set -e

format_folder () {
    folder=$1
    check=$2

    if [ $check = true ]
    then
        echo "Running format checking on folder: ${folder}"
        extra_flags='--dry-run --Werror'
    fi

    find $folder -iname *.h -o -iname *.cpp | xargs -I {} clang-format $extra_flags -i {}
}

check=false

while getopts 'c' flag; do
  case "${flag}" in
    c) check=true ;;
  esac
done

format_folder pt_soft_nms $check
format_folder tests $check
