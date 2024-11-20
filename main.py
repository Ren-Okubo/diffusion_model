import train_for_main
import test_for_main
import sys, argparse
if __name__ == '__main__':
    if sys.argv[1]: # sys.argv[1] is the bool about whether to train only or to train and test
        continue
    else:
        run_id, model_state_name = train_for_main.train_for_main()
        
    #datasetの分割
    #train
    #test
    
