#!/bin/sh
#  train_HOMF.sh
#  Created by Vatsal on 11/4/16.

for ptype in 'linear'
 do
    for ((window=2;window<=3;window=window+2))
     do
        for alpha in 1 
         do
            for lam in  0.0001
             do
                python HOMF.py -ptype $ptype -k 10 -maxit 2 -T $window -frac 100 -cg 20 -l $lam -train /Users/Vatsal/Google\ Drive/Vatsal/1m/Train.csv -val /Users/Vatsal/Google\ Drive/Vatsal/1m/Val.csv -alpha $alpha
             done
         done
     done
done
