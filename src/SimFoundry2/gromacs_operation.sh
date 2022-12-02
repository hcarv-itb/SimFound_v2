#!/bin/bash
#Created on Thu Jul 15 03:15:54 2021
#author: hcarv


#make index
gmx make_ndx -f sim1/acyl_octamer-eq.gro -o index_noW.ndx

#strip waters
for ((i=1;i<=10;i++))
    do gmx trjconv -f sim$i/$protein.xtc -o sim$i/${protein}_noW.xtc -n index_noW.ndx
    done