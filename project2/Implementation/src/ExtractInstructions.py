#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil  

#create testID for basic blocks (1 ID = 1 program)
#Split good/bad functions to different folder
def createTestID(block_dir, data_good, data_bad):
    testID = 0
    for folder in os.listdir(block_dir):
        if folder.endswith(".DS_Store"):
            continue
        myPath = os.path.join(block_dir , folder)
        for filename in os.listdir(myPath):
            source = os.path.join(myPath, filename)
            newpath = os.path.join(myPath , str(testID))
            if 'good' in filename:
                os.rename(source, newpath)
                shutil.move(newpath, data_good)  
            elif 'bad' in filename:
                os.rename(source, newpath)
                shutil.move(newpath, data_bad)  
            testID += 1
    print("Total functions in dataset: ", testID )
    
#Get all operands set 
def getALLOperands(path):
    op_set= set()
    for filename in os.listdir(path):
        if not filename.endswith('.DS_Store'):
            with open(path+ '/' + filename) as file:
                lines = file.read().splitlines()
                for line in lines:
                    strl = line.split('\t')
                    if len(strl) == 1:
                        continue
                    if len(strl) >2:
                        operands = strl[2].split(", ")#strl[2] = 'eax, dword ptr [rbp - 0x18]'
                        for myops in operands:
                            op_set.add(myops)   
    return op_set

#Get combined operands for feature extraction
def getCombinedOperands(path_all_operands):
    with open(path_all_operands, 'r') as reader:
        set_ops = reader.read().splitlines()
        data_type = eval(set_ops[0])
        reg_pointer = eval(set_ops[1])
        reg_type_64 = eval(set_ops[2])
        reg_type_32 = eval(set_ops[3])
        combine_type = data_type+reg_pointer+reg_type_64+reg_type_32
    return combine_type

def extractInstructionsAndOps(path, combine_type, addr = ['0x400', '0x500']):
    my_instr =[]
    set_filename= set()
    for filename in os.listdir(path):
        cnt = 0
        if not filename.endswith('.DS_Store'):
            with open(path+ '/' + filename) as file:
                insts_ops = []
                insts = ""
                lines = file.read().splitlines()
                for line in lines:
                    strl = line.split('\t')
                    if len(strl) == 1:
                        continue
                    insts = insts + strl[1] + " "#strl[1] = instruction like move, je
                    if len(strl) >2:
                        operands = strl[2].split(", ")#strl[2] = 'eax, dword ptr [rbp - 0x18]'
                        cnt = 1
                        for myops in operands:
                            isAddr_Reg = False
                            num = str("_op")+ str(cnt)
                            for op in combine_type:
                                if op in myops:
                                    insts = insts + (op+ num)  + " "
                                    isAddr_Reg = True
                                    continue
                            for ad in addr:
                                if ad in myops:
                                    insts = insts + ("address"+ num)  + " "
                                    isAddr_Reg = True
                                    continue
                            if isAddr_Reg == False:
                                if myops.isdigit() or '0x' in myops:
                                    insts = insts + ("constant" + num)  + " "
                            cnt +=1
                if not (insts.strip()) == "":
                    insts_ops.append(filename)           
                    insts_ops.append(insts.strip())
        set_filename.add(filename)
        my_instr.append(insts_ops) 
    return set_filename, my_instr

