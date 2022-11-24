#!/bin/sh 
"""
Created on Mon Aug 16 23:05:12 2021

@author: hcarv
"""

abc=("A" "B" "C" "D" "E" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V" "W" "X" "Y" "Z")

for ((i=1;i<=25;i++)); do 
    cp ./peptides/${pep}_X.pdb ./peptides/${pep}_$i.pdb
    done

for i in {0..25}; do 
    echo ${abc[$i]}; 
    find ./peptides/${pep}_$((i+1)).pdb -type f -exec sed -i "s/ X / ${abc[$i]} /g" {} \;
    done


for ((i=2;i<=25;i++)); do
    gmx insert-molecules -f SETD2_$pep.gro -ci ./peptides/${pep}_1.pdb -nmol 1 -o ./peptides/SETD2_${pep}_1.pdb
    i_prev=$((i-1)); gmx insert-molecules -f ./peptides/SETD2_${pep}_${i_prev}.pdb -ci ./peptides/${pep}_$i.pdb -nmol 1 -o ./peptides/SETD2_${pep}_$i.pdb -try 10000 -rot xyz 
    done
    
