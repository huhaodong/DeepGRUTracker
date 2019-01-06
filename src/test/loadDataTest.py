# import util.loadData as DataLoader
# import util.getArgs as getArgs
import sys
sys.path.append("F:\\work\\workspace\\DeepGRUTracker\\src")
import util.getArgs as getArgs
import util.loadData as DataLoader
import util.configLoader as configLoader
import os

if __name__=="__main__":

    cmdargs = getArgs.getCmdArgs()

    if cmdargs.config_path ==None or cmdargs=='None':
        pass
    else:
        if os.path.exists(cmdargs.config_path):
            cmdargs = configLoader.ConfigLoader(cmdargs.config_path)
        else:
            pass
    
    dataLoader = DataLoader.DataLoader(cmdargs)

    dataLoader.flashLoader()
    data = dataLoader.next()
    print('end')