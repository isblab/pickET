#! /bin/bash

for tomo_round_dir in `ls -d tomo*` ;
    do 
    echo $tomo_round_dir ; 
    cd $tomo_round_dir ; 
        for tomo_dir in `ls -d tomo_*` ;
            do
            cd $tomo_dir/coords ; 
                mv all_coords.ndjson all_annotations.ndjson
            cd ../.. ;
            done
    cd .. ; 
    done