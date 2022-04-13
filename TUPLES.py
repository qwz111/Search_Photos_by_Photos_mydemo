# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:35:39 2022

@author: admin
"""
import csv
import py2neo
from py2neo import Graph,Node,Relationship,NodeMatcher
g=Graph('http://localhost:7474',user='neo4j',password='qwz')
with open('tuples.csv','r',encoding='gbk') as f:
    reader=csv.reader(f)
    for item in reader:
        if reader.line_num==1:
            continue
        print("当前行数：",reader.line_num,"当前内容：",item)
        start_node=Node("Person",name=item[0])
        end_node=Node("Person",name=item[1])
      
        relation=Relationship(start_node,item[2],end_node)
        g.merge(start_node,"Person","name")
        g.merge(end_node,"Person","name")
        g.merge(relation,"Person","name")