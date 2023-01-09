import json
from elasticsearch import Elasticsearch
from datetime import datetime
#from LifeSeeker_API.ElasticsearchHelper.query_generator import QueryGenerator
#from LifeSeeker_API.ElasticsearchHelper.query_analyser import QueryAnalyser
#from LifeSeeker_API.utils.configs import *
#from LifeSeeker_API.Algorithms.Utils.datetime_utils import time_this

""" Configuration for Elasticsearch """
ELASTIC_HOST = 'localhost'
ELASTIC_PORT = 9222
ELASTIC_INDEX = 'vbs2023'

def time_this(func):
    def calc_time(*args, **kwargs):
        before = datetime.now()
        x = func(*args, **kwargs)
        after = datetime.now()
        print("Function {} elapsed time: {}".format(func.__name__, after-before))
        return x
    return calc_time

############# QUERY GENERATOR ##################
class QueryGenerator:
    def __init__(self): 
        self.ES_HOST = ELASTIC_HOST
        self.ES_PORT = ELASTIC_PORT
        self.INDEX = ELASTIC_INDEX

        self.INDEX_FIELDS = []
        self.MUST = []
        self.SHOULD = []
        self.FILTER = []

        es = Elasticsearch([{'host': self.ES_HOST, 'port': self.ES_PORT}])
        if es.ping():
            print("Connected to Elasticsearch node")
            mapping = es.indices.get_mapping(self.INDEX)
            fields = mapping[self.INDEX]['mappings']['properties']
            for field in fields:
                try:
                    if fields[field]['type']== 'text':
                        self.INDEX_FIELDS.append(field)
                except:
                    pass
            # print(mapping)
        else:
            print("Error: Cannot connect to Elasticsearch cluter")


    def gen_multi_matching_query(self, fields, values, optional=True, auto_fill=False):
        if auto_fill:
            weighted_fields = ["{0}^{1}".format(x, fields[x]) if x in fields else x for x in self.INDEX_FIELDS]
        else:
            weighted_fields = ["{0}^{1}".format(x, fields[x]) for x in fields]

        pattern = {"multi_match": {
            "query": values, 
            "fields": weighted_fields,
            "type": "most_fields" }}
        if optional:
            self.SHOULD.append(pattern)
        else: 
            self.MUST.append(pattern)


    def gen_matching_query(self, field, values, optional=True):
        pattern = {"match": {field: values}}
        if optional:
            self.SHOULD.append(pattern)
        else: 
            self.MUST.append(pattern)


    def gen_filtering_query(self, field, values):
        pattern = {"term": {field: values}}
        self.FILTER.append(pattern)


    def gen_match_all_query(self):
        pattern = {"match_all": {}}
        self.SHOULD.append(pattern)


    def reset_query(self):
        self.INDEX_FIELDS = []
        self.MUST = []
        self.SHOULD = []
        self.FILTER = []


    @time_this
    def run(self):
        query = {}
        bool_query = {}
        if len(self.MUST) > 0:
            bool_query["must"] = self.MUST
        if len(self.SHOULD) > 0:
            bool_query["should"] = self.SHOULD
        if len(self.FILTER) > 0:
            bool_query["filter"] = self.FILTER
        query["query"] = {"bool": bool_query}
        return query

    
################ QUERY ANALYSER #################
class QueryAnalyser:
    def __init__(self, query_generator):
        self.generator = query_generator

    def match_all(self, text_query, field):
        if len(text_query) == 0:
            self.generator.gen_match_all_query()
        else:
            fields_weight = {
                "ocr": 10,
                "color": 8
            }
            optionals = {
                "ocr": True,
                "color": False
            }
            # self.generator.gen_multi_matching_query(fields_weight, text_query, auto_fill=True)
            self.generator.gen_matching_query(field, text_query, optional=optionals[field])

            
    @time_this
    def analyse(self, text_query, mode="basic"):
        for field in text_query:
            self.match_all(text_query[field], field)
            
            
############## PROCESSOR ###################
class Processor:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.generator = QueryGenerator()
        self.analyser = QueryAnalyser(self.generator)

    def connect(self):
        self.es = Elasticsearch([{'host':self.host, 'port':self.port}])
        if self.es.ping():
            print("Connected to Elasticsearch node")
            return True
        else:
            print("Error: Cannot connect to Elasticsearch cluster")
            return False
        
    def update_data_field(self, doc_id, field, value):
        body = {
            "script": "ctx._source.{0} = {1}".format(field, value)
        }
        result = self.es.update(index=ELASTIC_INDEX, id=doc_id, body=body)
        return result


    # def get_testset_ranklist(self):
    #     self.generator.reset_query()
    #     self.generator.gen_matching_query("train_id", "test", optional=False)
    #     query = self.generator.run()
    #     result = self.es.search(index=ELASTIC_INDEX, body=json.dumps(query), size=10000)
    #     return(result)


    @time_this
    def search(self, text_query, mode="basic", continuous_query=False, size=10000):
        if not continuous_query:
            self.generator.reset_query()
        # text_query['ocr'], text_query['color']
        self.analyser.analyse(text_query, mode)
        query = self.generator.run()
        result = self.es.search(index=ELASTIC_INDEX, body=json.dumps(query), size=size)
        return result

es_processor = Processor(ELASTIC_HOST, ELASTIC_PORT)
es_processor.connect()
