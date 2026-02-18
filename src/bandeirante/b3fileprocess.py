import pandas as pd
import numpy as np
import os

import sqlite3 as sql
import json
import re

#import dask.dataframe as dd

Headers = [
    'TIPREG',    'DATA',    'CODBDI',    'CODNEG',    'TPMERC',    'NOMRES',
    'ESPEC',    'PRAZOT',    'MODREF',    'PREABE',    'PREMAX',    'PREMIN',
    'PREMED',    'PREULT',    'PREOFC',    'PREOFV',    'TOTNEG',    'QUATOT',
    'VOLTOT',    'PREEXE',    'INDOPC',    'DATVEN',    'FATCOT',    'PTOEXE',
    'CODISI',    'DISMES'
    ]

numericalHeaders = [
    'TIPREG',    'TPMERC',    'TOTNEG',    'QUATOT',    'INDOPC',    'DATVEN',    'FATCOT'    
]

moneyHeaders = [
    'PREABE',    'PREMAX',    'PREMIN',    'PREMED',    'PREULT',    'PREOFC',    'PREOFV',
    'VOLTOT',    'PREEXE',    'PTOEXE'
]

dropListArray = [
    'TIPREG',    'CODBDI',    #'CODNEG',    'TPMERC',    'NOMRES',    'ESPEC',    
    'PRAZOT',
        'MODREF',    #'PREABE',    #'PREMAX',    #'PREMIN',    #'PREMED',    #'PREULT',
        'PREOFC',    'PREOFV',    #'TOTNEG',    #'QUATOT',    #'VOLTOT',    #'PREEXE',
        'INDOPC',    #'DATVEN',    'FATCOT',
        'PTOEXE',    'CODISI',    'DISMES']

#este é o pattern de regex estabelecido pela própria bolsa
pattern = "(..)(........)(..)(............)(...)(............)(..........)(...)(....)(.............)(.............)(.............)(.............)(.............)(.............)(.............)(.....)(..................)(..................)(.............)(.)(........)(.......)(.............)(............)(...)"

#concatena tudo
def LeitorTabela(
        path,
        file,dropListOn = True,
        exportFile = True,
        dropList=dropListArray
        ):

    
    Dataset = pd.read_csv(path+"/" + file, encoding="latin1",dtype="string")
    
    #considerando o fato de que 
    if re.match(".+(.csv)",file) != None:
        return Dataset

    Dataset.drop(Dataset.tail(1).index, inplace=True) #la no final, fica um negocio que nem é dado de ação nem nada

    Dataset.columns = ["data"]
    Dataset = Dataset["data"].str.extract(pattern,expand=True)
    
    Dataset.columns = Headers


    Dataset["DATA"] = pd.to_datetime(Dataset["DATA"],format="%Y%m%d")
    Dataset[moneyHeaders+numericalHeaders] = Dataset[moneyHeaders+numericalHeaders].astype(np.int64) #isso pode ser otimizado

    Dataset[moneyHeaders] = Dataset[moneyHeaders]/100

    #dropa colunas desnecessárias

    if dropListOn:
        Dataset.drop(columns=dropList,inplace=True)
    
    #tira espaços adicionais
    Dataset["CODNEG"] = Dataset["CODNEG"].str.strip()
    Dataset["NOMRES"] = Dataset["NOMRES"].str.strip()
    
    Dataset.reset_index()      
    Dataset.sort_values(by=["CODNEG","DATA"],inplace=True)
    if exportFile:
        Dataset.to_csv(path+"/"+re.sub("\.(txt)|\.(TXT)","",file)+".csv")
    
    print("Processamento realizado: " + file)

    return Dataset


def DataUnion(
        path,
        name="correcteddata.csv",
        dropListOn = True,
        exportFile = True,
        concatFiles = False,
        dropList=dropListArray
        ):
    
    DatasetC = pd.DataFrame(columns=Headers)#pd.from_pandas(pd.DataFrame(columns=Headers))
    for file in os.listdir(path):
        Dataset = LeitorTabela(path,file, dropListOn, exportFile, dropList)
        if concatFiles:
            DatasetC = pd.concat([DatasetC, Dataset])#pd.from_pandas(Dataset,npartitions="auto")  ])
        print("Arquivo " + file + " carregado")
    
    #se vc quiser um CSV de tudo:
    if concatFiles:
        DatasetC.to_csv(name,index=False) 

    return DatasetC


def SQLLocal(
        databasePath = "../dados/", 
        dataPath = "../original/", 
        databaseName = "stock_market_data.db",
        alreadyRead = "alreadyRead.json",
        dropListOn = True,
        dropList=dropListArray):
    
    try:
        connection = sql.connect(databasePath + databaseName)
    except:
        with open(databasePath+databaseName,"w") as database:
            print(databaseName + " criado")


    try:
        with open(databasePath+alreadyRead) as readFile:
            alreadyReadDatabases = json.load(readFile)
    except:
        alreadyReadDatabases = []
        
    for arquivo in os.listdir(dataPath):
        if arquivo not in alreadyReadDatabases:
            Dataset = pd.read_csv(dataPath+"/" + arquivo, encoding="latin1",dtype="string")
                    
            Dataset.drop(Dataset.tail(1).index, inplace=True) #la no final, fica um negocio que nem é dado de ação nem nada

            Dataset.columns = ["data"]
            Dataset = Dataset["data"].str.extract(pattern,expand=True)
            
            Dataset.columns = Headers


            Dataset["DATA"] = pd.to_datetime(Dataset["DATA"],format="%Y%m%d")
            Dataset[moneyHeaders+numericalHeaders] = Dataset[moneyHeaders+numericalHeaders].astype(np.int64) #isso pode ser otimizado

            Dataset[moneyHeaders] = Dataset[moneyHeaders]/100

            #dropa colunas desnecessárias

            if dropListOn:
                Dataset.drop(columns=dropList,inplace=True)
            
            #tira espaços adicionais
            Dataset["CODNEG"] = Dataset["CODNEG"].str.strip()
            Dataset["NOMRES"] = Dataset["NOMRES"].str.strip()
            
            Dataset.reset_index()      
            Dataset.sort_values(by=["CODNEG","DATA"],inplace=True)
            Dataset.to_sql("cotacoes", connection, if_exists='append', index=False)
            
            alreadyReadDatabases.append(arquivo)
            
            print("Processamento realizado: " + arquivo)
        else:
            print("Processamento realizado: " + arquivo)

    with open(databasePath+alreadyRead, 'w', encoding='utf-8') as readFile:                
        json.dump(alreadyReadDatabases, readFile, ensure_ascii=False, indent=4)
            

    return connection

