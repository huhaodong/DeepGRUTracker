import os

def writerTrackerResult(data, outputPath, name):
    '''
        data: {frame:[[x,y,w,h]],frame2:[[x,y,w,h]]}
        name: name.txt
    '''
    path  = os.path.join(outputPath,name)
    print("in writer the path is:"+path)
    if os.path.exists(path):
        os.remove(path)
    with open(path,'a+') as outf:
        for k in data:
            ret = data[k].eval()
            for i in range(len(ret)):
                target = ret[i]
                if target[2]>0 and target[3]>0:
                    retS = str(k)+","\
                        +str(i+1)+","\
                        +str(target[0])+","\
                        +str(target[1])+","\
                        +str(target[2])+","\
                        +str(target[3])+",1,-1,-1,-1\n"
                    outf.write(retS)
                    outf.flush()