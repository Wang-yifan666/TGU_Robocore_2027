#!/bin/bash
cd "$(dirname "$0")/.."                       
cmake -S . -B build -DCMAKE_BUILD_TYPE=DeBUG 
cmake --build build -j$(nproc)