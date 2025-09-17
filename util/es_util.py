# @Time : 2020/6/19 10:09 上午 

# @Author : jiangtao16

# @File : es_util.py 

# @mail: jaingtao16@staff.weibo.com
import warnings
warnings.filterwarnings("ignore")
from elasticsearch import RequestsHttpConnection
from elasticsearch import Elasticsearch
import traceback
class ES:
    def __init__(self,log=None,use_new_es=True):
        if log:
            self.log = log
        else:
            from util import mylogging
            self.log = mylogging.get_logger('es_search')
        if use_new_es:

            es_address = [
            'https://jiangtao16:a7726022@push-aliyun-es-01.hebe.grid.sina.com.cn:9200',
            'https://jiangtao16:a7726022@push-aliyun-es-02.hebe.grid.sina.com.cn:9200',

              ]



            self.es_client = Elasticsearch(es_address, connection_class=RequestsHttpConnection, use_ssl=True,
                                           verify_certs=False)
        else:
            es_address = [
            'http://jiangtao16:a7726022@push-online-es-01.hebe.grid.sina.com.cn:10200',
            'http://jiangtao16:a7726022@push-online-es-02.hebe.grid.sina.com.cn:10200',
            'http://jiangtao16:a7726022@push-online-es-03.hebe.grid.sina.com.cn:10200'

              ]

            self.es_client = Elasticsearch(es_address, connection_class=RequestsHttpConnection, use_ssl=False,
                                           verify_certs=False)

        self.total = -1
        self.refresh_num = 0
        self.fresh = False
    def search(self,index_name,query,_source_include=None):
        ret = {}
        for i in range(3):
            try:
                ret = self.es_client.search(index=index_name, body=query, _source_includes=_source_include,request_timeout=130)
                break
            except:
                traceback.print_exc()
        self.total = ret.get('hits',{}).get('total',{}).get('value')
        docs = ret.get('hits',{}).get('hits',[])
        return docs
    def search_aggs(self,index_name,query,_source_include=None):
        ret = {}
        for i in range(3):
            try:
                ret = self.es_client.search(index=index_name, body=query, _source_includes=_source_include,request_timeout=130)
                break
            except:
                traceback.print_exc()
        #self.total = ret.get('hits',{}).get('total',{}).get('value')
        #docs = ret.get('hits',{}).get('hits',[])
        return ret
    def search_q(self,index_name,q,size=100,_source_include=None):
        ret = {}
        for i in range(3):
            try:
                ret = self.es_client.search(index=index_name, q=q, size=size,_source_includes=_source_include,
                                            request_timeout=130)
                break
            except:
                traceback.print_exc()
        self.total = ret.get('hits',{}).get('total',{}).get('value')
        docs = ret.get('hits',{}).get('hits',[])
        return docs
    def scroll(self,index_name,query,sort_atrrs = [],_source_include=None,include_id=False):
        ret = self.es_client.search(index=index_name, body=query,_source_includes=_source_include,request_timeout=300)
        docs = ret.get('hits',{}).get('hits',[])

        if self.total < 0:
            if isinstance(ret['hits']['total'],int):
                self.total = ret['hits']['total']
            else:
                self.total = ret['hits']['total']['value']
        while docs:
            #doc = None
            for doc in docs:
                if not include_id:
                    yield doc['_source']
                else:
                    yield doc
            query['search_after'] = doc['sort']
            ret = self.es_client.search(index=index_name, body=query,_source_includes=_source_include,request_timeout=300)
            docs = ret['hits']['hits']
    def delete_old_data(self,index_name,q):
        rt = self.es_client.delete_by_query(index=index_name,body={},q=q,wait_for_completion=False,conflicts='proceed',scroll_size=5000,slices=5)
        self.log.info('del_old_data:%s'%str(rt)[:150])
    def upsert_docs(self,es_cache):
        import json

        insert_ret = {'errors': True}
        for i in range(3):
            try:
                insert_ret = self.es_client.bulk(body=es_cache,request_timeout=100,refresh=True)
                break
            except:
                self.log.error(traceback.format_exc())
                self.log.error(es_cache)
        self.refresh_num += 1
        
        if insert_ret.get('errors') == True:

            self.log.info('es_push_ret@@@' + json.dumps(insert_ret, ensure_ascii=False))
        else:
            self.log.info('es_push_ret@@@' + json.dumps(insert_ret, ensure_ascii=False)[:200])
        failed = insert_ret.get('errors')
        return failed
if __name__ == '__main__':
    #from conf_py.conf import es_address

    es_address = ['http://10.182.13.51:10200',
                  'http://10.182.13.52:10200',

                  ]
    es_client = Elasticsearch(es_address)
    rt = es_client.search()
    print(rt)



