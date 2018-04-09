#!/bin/bash
make
rm *.o 


# # ------------------------------------------------------------------------------
# #  OptDigits
# # ------------------------------------------------------------------------------
# n=3823
# d=64
# m=2
# dname=OptDigits

# ./truth ${n} ${d} ${m} ${dname}.data ${dname}.cp2.0 ${dname}.fp2.0

# # ------------------------------------------------------------------------------
# #  Audio
# # ------------------------------------------------------------------------------
# n=54387
# d=192
# m=5
# dname=Audio

# ./truth ${n} ${d} ${m} ${dname}.data ${dname}.cp2.0 ${dname}.fp2.0

# # ------------------------------------------------------------------------------
# #  Color
# # ------------------------------------------------------------------------------
# n=67990
# d=32
# m=5
# dname=Color

# ./truth ${n} ${d} ${m} ${dname}.data ${dname}.cp2.0 ${dname}.fp2.0

# ------------------------------------------------------------------------------
#  OCR
# ------------------------------------------------------------------------------
n=3500000
d=1155
m=100
dname=OCR

./truth ${n} ${d} ${m} ${dname}.data ${dname}.cp2.0 ${dname}.fp2.0
