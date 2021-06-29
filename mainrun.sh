#!/bin/bash

g++-10 -fopenmp -I/usr/local/FFTW3/include -L/usr/local/FFTW3/lib main.cpp -lfftw3f -lfftw3f_threads -lm
