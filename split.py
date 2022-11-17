from random import randint
import math

def main():
    file = open('train.csv', "r")
    train = open('newtrain.csv', "w")
    test = open('test.csv', "w")


    readfile = [line for line in file]
    firstline = readfile.pop(0)
    
    count = 0
    total = math.floor(len(readfile) * .8)


    # Write 80 % to train
    train.write(firstline)
    while count < total:
        index = randint(0, len(readfile) - 1)
        train.write(readfile.pop(index))
        count += 1 
    
    # Write remaining to test
    test.write(firstline)
    while readfile:
        test.write(readfile.pop(0))
    
    print("FINISHED")

    return

main()