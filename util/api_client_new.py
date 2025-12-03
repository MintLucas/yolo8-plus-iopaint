#coding=utf-8
import json
import re

import requests,traceback
import sys,os
sys.path.append(os.getcwd())

import time
from util.token_util_new import token_fresh
from util.mylogging import get_logger
class Api_client:
    def __init__(self,logger = get_logger("util_log/Api_client"),session=None):
        self.session = requests.session() if not session else session
        self.log = logger
        self.token_help = token_fresh(log=self.log, session=self.session)
        self.source = '2552521548'
        self.time_now = 0
    def get_header(self,uid):
        use_source = self.source
        s_url = f"http://ip1.push.weibo.cn:9033/get_uid_token?source={use_source}"
        if uid:
            s_url =  f"http://ip1.push.weibo.cn:9033/get_uid_token?source={use_source}&ips=10.23,192,14&uid={uid}"
        headers = {}
        try:
            r = self.session.get(s_url)
            js = r.json()
            if 'data' in js and js['data']:
                headers['Authorization'] = js['data']
                headers['API-RemoteIP'] = js['ip']
        except:
            self.log.warning(traceback.format_exc())
        #print(headers)
        return headers

    def get_showbatch(self,mids,header):
        if isinstance(mids,list):
            mids_str = ','.join(mids)
        else:
            mids_str = mids
        url = 'http://i.media.api.weibo.com/2/video/show_batch.json?source=2552521548&id_type=0&part=origin,playback,statistics,playlists,author,activities,contribution,status,current_status&ids=' + mids_str
        js = {}
        try:
            r = self.session.get(url,headers=header,timeout=3)
            js = r.json()
        except:
            er_msg = traceback.format_exc()
            self.log.error(f'err show batch:{er_msg}')
        #url = 'http://i2.api.weibo.com/2/statuses/show_batch.json?source=356732087&isGetLongText=1&simplify=1&ids=' + mids_str

        return js

    def post_api(self,url,method='post',data = None,timeout=3,uid=None,param = None,is_byte=False):
        now = time.time()
        if now - self.time_now > 60*10 or uid:
            self.time_now=now
            #50秒更新一次token
            temp_head = self.get_header(uid)
            if temp_head:
                self.headers = temp_head
            # 50秒更新一次在线广告位信息

        #print(self.headers)
        #print(self.headers)
        ret = None
        try:
            if method == 'post':
                r = self.session.post(url,headers = self.headers,data=data,timeout = timeout)
            else:
                dd = {'url':url,'headers':self.headers,'timeout':timeout}
                if param:
                    dd['params'] = param
                r = self.session.get(**dd)
            if is_byte:
                ret = r.content
            else:
                ret= r.json()
        except:
            er_msg = traceback.format_exc()
            #self.log.error(r.content)
            self.log.error(f'{url} {er_msg}')
            if is_byte:
                ret = ''
            else:
                ret = {}
        return ret

    def explain_long(self,short_url):
        url = 'http://i.api.weibo.com/2/short_url/expand.json'
        par = {'source':self.source,'url_short':short_url}
        data = self.post_api(url,method='get',param=par)
        #print(data)
        url_short = ''
        try:
            url_short = data['urls'][0]['url_long']
        except:
            pass
        return url_short

    def get_mid_text(self, mid, uid = None):
        from jsonpath import jsonpath
        try:
            header = self.token_help.get_token(source='356732087',uid=uid)
            js = self.get_showbatch(mid,header)
        except:
            er_msg =traceback.format_exc()
            self.log.error(f'{er_msg}')
        import re
        self.log.info('start get_mid_text')
        txt = ''
        try:
            txts = jsonpath(js,f'$.{mid}.status_info.text')
            if isinstance(txts,bool):
                txts = jsonpath(js,f'$.{mid}.current_status_info.text')
            txt = txts[0]
        except:
            er_msg = traceback.format_exc()
            self.log.error(f'call show batch error:{er_msg}')
        self.log.info(f'get_mid_text:{txt}')
        if txt:
            txt = re.sub('http[:/\.\w]+','',txt)
            self.log.info(f'clear mid_text:{txt}')
        return txt
    
    
    def get_normal_mid(self,mid_source):
        mid_source = mid_source.split('/')[-1]
        if re.match('^http://weibo.com/[0-9]+/[A-Za-z0-9]+$',str(mid_source)):
            mid_source = mid_source.split('/')[-1]
        #url = 'http://i.api.weibo.com/2/statuses/queryid.json'
        url = f"http://i.api.weibo.com/2/statuses/queryid.json?source=2552521548&mid={mid_source}&type=1"
        #print(url)
        param = {'source':'356732087','mid':mid_source,'type':1,'isBase62':1}
        url_data = self.post_api(url,method='get',param=param)
        mid = None
        try:
            mid = url_data.get('id')
        except:
            self.log.error(f'urldata:{url_data}')
        return mid

if __name__ == '__main__':
    api_clinet = Api_client()
    test = api_clinet.get_mid_text("5208696615342390")
    print(test)
