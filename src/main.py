import util.getArgs as getArgs
import trainDeepGRUTracker as train
import testDeepGRUTracker as test
import util.configLoader as configLoader
import os

if __name__=='__main__':
    cmdargs = getArgs.getCmdArgs()

    if cmdargs.config_path ==None or cmdargs=='None':
        pass
    else:
        if os.path.exists(cmdargs.config_path):
            cmdargs = configLoader.ConfigLoader(cmdargs.config_path)
        else:
            pass

    if cmdargs.is_train:
        print("in train! is_train={}".format(cmdargs.is_train))
        train.train(cmdargs)
    else:
        print("in test! is_train={}".format(cmdargs.is_train))
        test.test(cmdargs)