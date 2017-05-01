import main_ovo
import os

star_seperator = '*'*100
print star_seperator
print "                                         TEAM NO : 65"
print star_seperator
print "         Reduction of Multiclass classfication into one vs one binary classification"
print star_seperator
algo_opt = 0
while algo_opt != 5 :
    print "\nMENU: \nChoose an algorithm to run: \n 1: One vs one (OVO)\n 2: Cost sensitive one vs one (CSOVO)\n 3: Wiegthed all pairs (WAP)\n 4: Cost weighthed Neural network (CWNN)\n 5: Exit\n\n"
    print " Enter your option: \n"
    algo_opt = input()
    if algo_opt == 1 :
        execfile("main_ovo.py")
    elif algo_opt == 2 :
        execfile("main_csovo.py")
    elif algo_opt == 3 :
        execfile("main_wap.py")
    elif algo_opt == 4 :
        execfile("main_cwnn.py")
    elif algo_opt == 5 :
        exit()
    else :
        print "Choose a correct option\n"