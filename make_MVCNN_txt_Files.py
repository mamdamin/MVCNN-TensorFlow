
# coding: utf-8

# #### Read png file pathes and make txt files for MV_CNN code

# In[1]:


import sys
#import numpy as np
import glob
#import re
#from PIL import Image
import random
import pandas as pd
import os


RootFolder = sys.argv[1]#'/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views'
subjects   = os.listdir(RootFolder)

airwayPhenotypes = pd.read_csv('/Users/amotahari/Data/airway_phenotypes_for_fgf10_replication.csv',index_col=0)
cleanup_nums = {"Lower_lobe_segmental_status":     {"Standard": 0, "Acc. B*": 1, "Abs. RB7": 2}}

#print(airwayPhenotypes.head(2))
#sid = intersection(subjects,airwayPhenotypes.index.tolist())
#print(len(sid))
print("Subjects with airway classification: {}".format(len(airwayPhenotypes)))
print("Subjects with airway mask: {}".format(len(subjects)-2))

images = glob.glob(RootFolder+'/**/*.png')
airwayPhenotypes = airwayPhenotypes.loc[airwayPhenotypes.index.intersection(subjects)]
print(airwayPhenotypes.Lower_lobe_segmental_status.value_counts())

airwayPhenotypes.replace(cleanup_nums, inplace=True)
subjects = airwayPhenotypes.index.tolist()

# #### Bundle png images in txt file for use in MV-CNN

# In[7]:


#order = range(48)#[0,1,4,5,6,7,8,9,10,11,2,3]
df = airwayPhenotypes.loc[:,['Lower_lobe_segmental_status']]

try:
    os.mkdir(os.path.join(RootFolder,'txtImages'))
except:
    print('txtImages already exists')
try:
    os.mkdir(os.path.join(RootFolder,'sets'))
except:
    print('sets already exists')

    #df.rename(index=str, columns={"A": "a"
for subject in subjects:
    folder = os.path.join(RootFolder,subject)
    images = glob.glob(folder+'/*.png')
    images.sort()
    pre = [1, len(images)]
    pre = [airwayPhenotypes.loc[subject,'Lower_lobe_segmental_status'], len(images)]
    filename   = subject+'_Cat_{}.txt'.format(pre[0])
    thefile = open(os.path.join(RootFolder,'txtImages',filename), 'w')
    images = pre+images#[ images[i] for i in order]
    for item in images:
        thefile.write("%s\n" % item)
    thefile.close()
    df.loc[subject,'txtImages'] = os.path.join(RootFolder,'txtImages',filename)
    #print(subject)


# In[8]:


#print(df.head(2)) 


# In[9]:


random.seed(0)
#random.shuffle(images)
images = df.sample(frac=1,random_state=1)
#print(images.head(2))


# #### Make list of images for Train, Test and Validation

# In[10]:


n = len(images)
print("Number of subjects taht have airway type and mask: {}".format(n))

sets = {'train':images[:int(n*.6)],
        'validation':images[int(n*.6):int(n*.8)],
        'test':images[int(n*.8):]}

#print(images.Lower_lobe_segmental_status.values)

c=1
for setName, imageList in sets.items():
    thefile = open(os.path.join(RootFolder,'sets',setName+'.txt'), 'w')
    #print(setName,imageList.shape)
    print("{} set statistics: \n{}\n".format(setName, imageList.Lower_lobe_segmental_status.value_counts()))
    for sid, item in imageList.iterrows():
        line = '{} {}\n'.format(item.txtImages, 1)
        line = '{} {}\n'.format(item.txtImages, item.Lower_lobe_segmental_status)
        #print(line)
        thefile.write(line)
    thefile.close()
   


# In[11]:


print(sid)
