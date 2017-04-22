import os, sys

input = open(sys.argv[1], 'r')
output = open('log.txt', 'w')

lines = input.readlines()
for line in lines:
    if line[0:9] != '[mpeg4 @ ':
        output.write(line)
        
output.close()
input.close()