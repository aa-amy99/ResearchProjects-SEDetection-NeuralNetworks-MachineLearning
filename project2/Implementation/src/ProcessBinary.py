import angr
import sys
import os
import errno
import shutil  


def createProject (myPath): 
    proj = angr.Project(myPath)
    return proj


def getAllBasicBlocks(proj, blockPath, projName, keyName):
    cfg = proj.analyses.CFGFast()
    funcAddr = 0x400000 #for Main Function only
    for f in proj.kb.functions.values():
        for key in keyName:
            if (key in f.name):
                funcAddr = f.addr
                cfg = proj.analyses.CFGFast()
                func = cfg.kb.functions[funcAddr]
                print(f.name, funcAddr)
                f = open(blockPath + '/' + f.name[-20:] , 'w')
                original = sys.stdout
                sys.stdout = f
                for b in func.blocks:
                    print(b)
                    b.pp()
                sys.stdout = original
                f.close()
                print("done function/n")

def processProgram(block_dir, dir, filename, keyName ):
    myPath = os.path.join(dir, filename)
    projName = os.path.splitext(filename)[0]
    blockPath = os.path.join(block_dir, projName[0:250])
    if not os.path.exists(blockPath):
      try:
        os.makedirs(blockPath, 0o700)
      except OSError as e:
        if e.errno != errno.EEXIST:
          raise
      if not blockPath.endswith('.DS_Store'):
          proj = createProject(myPath)
          getAllBasicBlocks(proj, blockPath, projName,keyName)

def generateBlocksFromBinary(raw_dir, block_dir):
    keyName = ['good', 'bad']
    cnt = 0
    for folder in os.listdir(raw_dir):
        if folder.endswith(".DS_Store"):
            continue
        myPath = os.path.join(raw_dir, folder)
        for filename in os.listdir(myPath):
            processProgram(block_dir, myPath, filename, keyName)
            cnt = cnt +1
    print("Total binary programs processed: ", cnt)
