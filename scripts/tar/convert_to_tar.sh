#!/bin/bash

cd ../../data-formats/raw/tiny-imagenet-200
for dir in */; do
    DIR=$(basename "$dir")
    
    echo "Working on $DIR"
    tar -cf "$DIR.tar" -C "$DIR" .

    mv $DIR.tar ../../tar
    
done
