import util.getArgs as getArgs
import trainDeepGRUTracker as train

if __name__=='__main__':
    cmdargs = getArgs.getCmdArgs()
    if cmdargs.is_train:
        train.train(cmdargs)
    else:
        pass