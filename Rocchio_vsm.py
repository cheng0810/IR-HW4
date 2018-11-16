
# coding: utf-8




import numpy as np
import pandas as pd
import math 
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm



"""
1st vsm to find out top n documents
"""

#cal all vocabulary
with open('query_list.txt') as file:
    query_list = file.read().rstrip().split()
with open('doc_list.txt') as file:
    doc_list = file.read().rstrip().split()

all_voc = []
query_voc = []
doc_voc = []
for query_name in query_list:      #cal query_voc
    with open('Query/' + query_name) as file:
        voc = file.read().replace('-1','').rstrip().split()
        query_voc.extend(voc)

for doc_name in doc_list:      #cal doc_voc
    with open('Document/' + doc_name) as file:
        for line in range(3):  #we don't want first three line data
            file.readline() 
        voc = file.read().replace('-1','').rstrip().split()
        doc_voc.extend(voc)
query_voc = list(set(query_voc))
doc_voc = list(set(doc_voc))

all_voc.extend(query_voc)
all_voc.extend(doc_voc)
all_voc = list(set(all_voc))   #throw the same voc


#print(len(all_voc))   #check total voc amout


#cal query term frequency
query_TF = np.zeros((len(all_voc),len(query_list)))

for query_num,query_name in tqdm(enumerate(query_list)):
    with open('Query/' + query_name) as file:
        file = file.read().replace('-1','').rstrip().split()
        for word in range(len(all_voc)):
            if(file.count(all_voc[word])) >0:
                query_TF[word,query_num] = 1 + math.log(file.count(all_voc[word]),2)
            else:
                query_TF[word,query_num] = 0


#cal document term frequency and IDF
doc_TF = np.zeros((len(all_voc),len(doc_list)))
voc_IDF = np.zeros((len(all_voc),1))

for doc_num,doc_name in tqdm(enumerate(doc_list)):     #doc TF
    with open('Document/' + doc_name) as file:
        for line in range(3):  #we don't want first three line data
            file.readline() 
        file = file.read().replace('-1','').rstrip().split()
        for word in range(len(all_voc)):
            if(file.count(all_voc[word])) >0:
                doc_TF[word,doc_num] = 1 + math.log(file.count(all_voc[word]),2)
                voc_IDF[word] += 1 
            else:
                doc_TF[word,doc_num] = 0
                
for word in range(len(all_voc)):     #all voc IDF
    if(voc_IDF[word]) >0:
        voc_IDF[word] = math.log(2265/voc_IDF[word],10)



#query and document TFxIDF
query_TFIDF = np.zeros((len(all_voc),len(query_list)))
doc_TFIDF = np.zeros((len(all_voc),len(doc_list)))

for num in range(len(query_list)):
    for word in range(len(all_voc)):
        query_TFIDF[word,num] = query_TF[word,num]*voc_IDF[word]
        
for num in range(len(doc_list)):
    for word in range(len(all_voc)):
        doc_TFIDF[word,num] = doc_TF[word,num]*voc_IDF[word]

#first time cos() and put the result into pandas
VSM = np.zeros((len(query_list),len(doc_list)))

for q_num in tqdm(range(len(query_list))):
    for d_num in range(len(doc_list)):
        VSM[q_num,d_num] = cosine_similarity([query_TFIDF[:,q_num]],[doc_TFIDF[:,d_num]])

VSM = pd.DataFrame(VSM,columns = doc_list ,index = query_list)
#VSM

top = 20   #the top n document to fix the query
alpha = 0.3
beta = 0.9

q_TFIDF = np.zeros((len(all_voc),len(query_list)))     #new query_TFIDF
q_TFIDF = query_TFIDF


for num in tqdm(range(len(query_list))):   #update query TFIDF
    VSM = VSM.sort_values(by = query_list[num],ascending= False,axis = 1) #sort value
    cols = list(VSM.columns.values) #take the sorted list
    Doc_TFIDF = pd.DataFrame(doc_TFIDF,columns = doc_list,index = all_voc)
    Doc_TFIDF = Doc_TFIDF[cols] #取排序後的col順序
    Doc_TFIDF = Doc_TFIDF.iloc[:,:top] #取前n個doc
    Doc_TFIDF = Doc_TFIDF.values
    rq = np.sum(Doc_TFIDF,axis = 1)
    q_TFIDF[:,num] = alpha*q_TFIDF[:,num] + beta*(rq/top)

#second time cos() and put the result into txt
VSM2 = np.zeros((len(query_list),len(doc_list)))

for q_num in tqdm(range(len(query_list))):
    for d_num in range(len(doc_list)):
        VSM2[q_num,d_num] = cosine_similarity([q_TFIDF[:,q_num]],[doc_TFIDF[:,d_num]])

VSM2 = pd.DataFrame(VSM2,columns = doc_list ,index = query_list)

f = open('rocchio.txt','w')    #write in txt file
f.write('Query,RetrievedDocuments\n')
for i in range(len(query_list)):
    f.write(VSM2.index[i])
    f.write(',')
    VSM2 = VSM2.sort_values(by = query_list[i],ascending= False,axis = 1)
    for j in range(len(doc_list)): #this time evaluate jusy map@50
        f.write(VSM2.columns[j])
        f.write(' ')
    f.write('\n')


# f = open('testvsm.txt','w')    #write in txt file
# f.write('Query,RetrievedDocuments\n')
# for i in range(len(query_list)):
#     f.write(VSM.index[i])
#     f.write(',')
#     VSM = VSM.sort_values(by = query_list[i],ascending= False,axis = 1)
#     for j in range(len(doc_list)): #this time evaluate jusy map@50
#         f.write(VSM.columns[j])
#         f.write(' ')
#     f.write('\n')

