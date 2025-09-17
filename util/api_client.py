#coding=utf-8
import json
import re

import requests,traceback

import time

class Api_client:
    def __init__(self,logger,session=None):
        self.session = requests.session() if not session else session
        self.source = '2552521548'
        ##私信聊天业务，要求实时获取用户短期兴趣特征，合理使用用户短期兴趣特征，回复用户聊天内容
        self.token_url = f'http://10.30.61.53:8122/get_token_header?source={self.source}'

        self.log = logger
        self.time_now = time.time()
        self.headers = self.get_header(None)
        #self.baidu_balck_term ={}
        #self.get_value_from_applo('baidu_black_term')

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
    def get_normal_mid(self,mid_source):
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
    def download_message_data(self,att_id,uid):
        import base64
        url = f'http://i.upload.api.weibo.com/2/mss/msget?source={self.source}&fid={att_id}'
        print(url)
        ret = self.post_api(url,method='get',uid=uid,is_byte=True)
        print(ret)
        #二进制数据一般很长 太短  则认为请求失败
        b64= None
        if ret and len(ret)>=150:
            self.log.info(f'base64:{str(ret)[:90]}')
            b64 = base64.b64encode(ret).decode()
        return b64

    def analyst_img(self,b64):
        url = 'http://llm-yongfeng.multimedia.wml.weibo.com/mm-wb-ml-push/qwen2-vl-weibo-bigdata-push/v2/models/qwen2-vl/generate'
        pic_url=f"data:image/png;base64,{b64}"
        data1 = {
                "urls": [
                    #'https://wx1.sinaimg.cn/orj360/008CmtQnly1hvutafdi1oj30rk11edk5.jpg'
                    pic_url
                         ],
                "text": "请描述图片内容",
                "type": "image_base64"
            }
        ret = self.post_api(url,data=json.dumps(data1),timeout=20)
        ans = ''
        try:
            ans = ret['result']
        except:
            er_msg = traceback.format_exc()
            self.log.error(f'{er_msg}')
        return ans



    def get_chat_days(self,role_uid,uid):
        url = f"http://i2.api.weibo.com/2/direct_messages/continuous_chat_tags.json?source={self.source}&uid={uid}"
        self.headers = self.get_header(role_uid)
        para = {"source": self.source,
                    "uid": uid,
                }
        url_data = self.post_api(url=url,uid=role_uid,param=para,method='get')
        return url_data.get('days',0)



    def get_tts_result(self,base64_str):
        url = "http://llm-yongfeng.multimedia.wml.weibo.com/mm-wb-ml-push/gptsovits-weibo-bigdata-push-v2/v2/models/gptsovits/infer"
        payload = {
        "id": "1111",#这个id可以随便填，用来区分不同的请求
        "inputs": [
            {
                "shape": [1],
                "datatype": "BYTES",
                "name": "input_parameters",
                "data": [ "{\"task\":\"asr\",\"tasktype\":\"sensevoice\",\"audio_b64\":\"" + base64_str + "\",\"text_lang\":\"all_zh\"}"]
            }
        ]
    }
        end_result = ''
        cs = ''
        try:
            r = self.session.post(url,json=payload,timeout=30)
            cs = r.content.decode()
            js = r.json()
            data = js['outputs'][0]['data']
            data = data[0]
            data_js = json.loads(data)
            end_result = data_js['data']
        except:
            er_msg = traceback.format_exc()
            self.log.error(f'cs:{cs} {er_msg}')


        return end_result
    def get_tag_cn(self,tags):
        url = f"http://i.feature.weibo.com/feature/object/query.json?source=356732087&fids=622&oids=" + tags
        data  = self.post_api(url,method='get')
        data_use = data['data']
        end = {tag_id:vmp.get('622') for tag_id,vmp in data_use.items() if vmp.get('622')}
        return end
    def get_mid_tags_much(self, mid):
        mid = str(mid)
        help_vmp = {'10419':'cat','10482':'cat',
                    '10420':'tag','10483':'tag',
                    '10421':'obj','10484':'obj'
                    }
        url = "http://i.feature.weibo.com/feature/mblog/query.json?source=356732087&mids=" + mid + '&fids=10419,10420,10421,10482,10483,10484'
        end_ret = {}
        res = self.post_api(url)
        data = res.get('data',{})
        #print(data)
        use_tags = set()

        for one_mid,vmp in data.items():
            vmp1 = {}
            for fid,vals in vmp.items():
                ss = vals.split('|')

                pairs = [x.split('@') for x in ss if len(x.split('@'))==2]
                pairs = [x[0] for x in pairs if float(x[1])>=0.5]
                use_tags|=set(pairs)
                ch_fid = help_vmp[fid]
                if pairs:
                    vmp1[ch_fid] = list(set(pairs))
                #fid2tag_ids[fid] = list(set(pairs))
            if vmp1:
                end_ret[one_mid] = vmp1
        use_tags = list(use_tags)
        parts = [use_tags[x:x+30] for x in range(0,len(use_tags),30)]
        all_id2cn = {}
        for part in parts:
            if not part:
                break
            _id2cn = self.get_tag_cn(','.join(part))
            all_id2cn.update(_id2cn)

        for one_mid_1 ,vvmp in end_ret.items():

            bad_ids = set()
            for fid_1 in vvmp.keys():
                ss = [all_id2cn[x] for x in vvmp[fid_1] if all_id2cn.get(x)]
                vvmp[fid_1] = ss
                if not ss:
                    bad_ids.add(fid_1)
            for b in bad_ids:
                vvmp.pop(b)
        return end_ret



        #print(all_id2cn)
    def get_user_tags(self,uid,weight=0.45,top_k = None):
        url = f'http://i.feature.weibo.com/feature/user/query.json?source={self.source}&uids={uid}&fids=21177,21178,21179'
        #print(url)

        url_data = self.post_api(url, method='get')
        tags = url_data['data'][uid]
        use_tags = set()
        fid2tag_ids = {}
        for fid,vals in tags.items():
            ss = vals.split('|')
            pairs = [x.split('@') for x in ss if len(x.split('@'))==2]
            pairs = [(x[0],float(x[1])) for x in pairs if float(x[1])>=weight]
            if top_k:
                pairs.sort(key=lambda stu:stu[1],reverse=True)
                pairs = pairs[:top_k]
            pairs = [x[0] for x in pairs]
            use_tags|=set(pairs)
            fid2tag_ids[fid] = list(set(pairs))
            #print(fid,pairs)
        #print(use_tags)
        _id2cn = self.get_tag_cn(','.join(use_tags))
        for k in fid2tag_ids.keys():
            ids = fid2tag_ids[k]
            cns = [_id2cn[x] for x in ids if _id2cn.get(x) ]
            fid2tag_ids[k]=cns
        return fid2tag_ids

    def get_comment_by_me(self,uid,page,count):
        url = f'http://i.api.weibo.com/2/comments/by_me.json?source={self.source}'
        data = self.post_api(url,uid=uid,method='get')
        return data
    def get_uid_show_mul(self,uids):
        uid_use = uids if isinstance(uids,str) else ','.join(uids)
        url = f'https://i.api.weibo.com/users/show_batch.json?source=356732087&uids={uid_use}'
        data = self.post_api(url,method='get')
        ret = {}
        for ucom in data.get('users',[]):
            if ucom.get('idstr') and ucom.get('status'):
                ret[str(ucom.get('idstr'))] = ucom['status']
        return ret
        pass
    def get_uid_show_batch(self,uid):
        url = f"http://i2.api.weibo.com/users/show.json?source=356732087&uid={uid}"

        result_dict = {}

        try:

            data = self.post_api(url,method='get')
            uid = data.get("id", None)
            if uid:
                result_dict[str(uid)] = data
        except Exception as e:
            er_msg = traceback.format_exc()
            self.log.error(er_msg)
        return result_dict

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
    def get_audio_text(self, media_id):
        url = 'http://i.media.api.weibo.com/2/media/asr.json?source=99075054&media_id=' + media_id
        url_data = self.post_api(url, method='get')
        return url_data
    def get_mid_feature_ag(self,mids):
        if isinstance(mids,list):
            mids_str = ','.join(mids)
        else:
            mids_str = mids
        url = 'http://i2.api.weibo.com/2/statuses/show_batch.json?source=356732087&isGetLongText=1&simplify=1&ids=' + mids_str
        url_data = self.post_api(url, method='get')
        return url_data
        #import json
        #return json.dumps(url_data,ensure_ascii=False)
        rets = {}
        import re
        try:
            for one in url_data['statuses']:
                mid = one['idstr']
                text = one.get('longText',{}).get('longTextContent') if one.get('longText',{}).get('longTextContent') else one.get('text','')
                if one.get('deleted')=='1' or '此微博已被删除' in text:
                    text= ''
                text = text[:800]
                if not text:
                    continue
                vmp = {
                    'uid':one.get('user',{}).get('idstr',''),
                    'blog':text
                }
                rets[mid]=vmp
        except:
            import traceback
            err_msg = traceback.format_exc()
            self.log.error(err_msg)
        #print(json.dumps(url_data,ensure_ascii=False))
        return rets
    def get_mid_feature(self,mids):
        if isinstance(mids,list):
            mids_str = ','.join(mids)
        else:
            mids_str = mids
        url = 'http://i2.api.weibo.com/2/statuses/show_batch.json?source=356732087&isGetLongText=1&simplify=1&ids=' + mids_str
        url_data = self.post_api(url, method='get')
        #print(url_data)
        #return url_data
        #import json

        #return json.dumps(url_data,ensure_ascii=False)
        rets = {}
        import re
        try:
            for one in url_data['statuses']:
                mid = one['idstr']
                text = one.get('longText',{}).get('longTextContent') if one.get('longText',{}).get('longTextContent') else one.get('text','')
                if one.get('deleted')=='1' or '此微博已被删除' in text:
                    text= ''
                text = text[:800]
                source = str(one.get('source',''))
                source=re.sub('<[^>]+>','',source)
                vmp = {
                    'uid':one.get('user',{}).get('idstr',''),
                    'blog':text,
                    'user_name':one.get('user',{}).get('name',''),
                    'verified_type':one.get('user',{}).get('verified_type',''),
                    'verified_reason':one.get('user',{}).get('verified_reason',''),
                    'user_description':one.get('user',{}).get('description',''),
                    'followers_count':one.get('user',{}).get('followers_count',0),
                    'source_type':one.get('source_type',''),
                    'pic_num':one.get('pic_num',0),
                    'source':source,
                    'reposts_count':one.get('reposts_count',0),
                    'comments_count':one.get('comments_count',0),
                    'attitudes_count':one.get('attitudes_count',0)

                }
                rets[mid]=vmp
        except:
            import traceback
            err_msg = traceback.format_exc()
            self.log.error(err_msg)
        #print(json.dumps(url_data,ensure_ascii=False))
        return rets
    def get_ocr(self):
        #https://wx4.sinaimg.cn/mw690/006tCxXily1hjsd0d2vfjj31hb1z3diy.jpg
        api_url = 'http://i.feature.weibo.com/feature/get_pid.json?source=356732087&fids=%s&pids=%s' % (1198, '007BD2ERly1hjsf86njbnj31hc0u0wh6')

        js_data = self.post_api(api_url,method='get')
        #print(js_data)
        import json
        #print(json.dumps(js_data,ensure_ascii=False,indent=4))
        return js_data
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
    def get_short_url(self,url_long):
        url = 'http://i.api.weibo.com/2/short_url/shorten.json'
        par = {'source':self.source,'url_long':url_long}
        data = self.post_api(url,method='get',param=par)
        url_short = ''
        try:
            url_short = data['urls'][0]['url_short']
        except:
            pass
        return url_short


    def object_showbatch(self):
        '''
        photo.weibo.com/h5/comment/compic_id/1022:2305971c74958b45c78bdc0f52b63728061eda,topic_id=>

        '''
        url = 'http://i.api.weibo.com/object/show_batch.json?source=356732087&object_ids=1034:5061962262839356'
        rt = self.post_api(url,method='get')
        import json
        #print(json.dumps(rt,ensure_ascii=False))
    def cid_showbatch(self,cid):
        url = f"http://i2.api.weibo.com/2/comments/show_batch.json?source=356732087&cids={cid}"
        #print(url)
        data = self.post_api(url,method='get')
        #print(json.dumps(data,ensure_ascii=False))
        return data
    def get_mid_tag(self,mids):
        mids_str = mids if isinstance(mids,str) else ','.join(mids)
        url = "http://i.feature.weibo.com/feature/mblog/query.json?source=356732087&mids=" + mids_str + '&fids=10419,10482'
        data = self.post_api(url)
        return data
    def get_uid_feature(self,uids=None):
        '''
        {
      "mid": "4891668127680151",
      "blog": "欧阳娜娜竟然真的把黄宗泽吃剩下的那碗面全吃光了，看得出来是真没洁癖啊[允悲]",
      "uid": "1767076672",
      "firsttags":"",
      "user_name": "小影娱乐",
      "verified_type": 0,
      "verified_reason": "微博电视团成员 娱乐博主",
      "user_description": "专注明星娱乐；带你吃最新的瓜，看最猛的料！",
      "followers_count": 2082259,
      "图片数量": 2,
      "source": "Android客户端",
      "source_type": 1
    }
        :param uids:
        :return:
        '''
        if isinstance(uids,list):
            uids_str = ','.join(uids)
        else:
            uids_str = uids

        url = f'http://i.api.weibo.com/users/show_batch.json?source=356732087&uids={uids_str}'
        js=self.post_api(url,method='get')
        import json
        #print(json.dumps(js,ensure_ascii=False,indent=4))
        rets = {}
        import re
        try:
            for one in js['users']:
                uid = one['idstr']
                #soure = one.get('source','')
                #soure=re.sub('<[^>]+>','',soure)
                vmp={
                    'user_name':one.get('screen_name',''),
                    'verified_type':one.get('verified_type',-1),
                    'verified_reason':one.get('verified_reason',''),
                    'user_description':one.get('description',''),
                    'followers_count':one.get('followers_count',0),
                    'verified':one.get('verified'),
                    #'source':soure,
                    'source_type':one.get('source_type')

                }
                rets[uid] = vmp

        except:
            pass
        return rets
    def parse_image_content(self, image_url, prompt="描述图片", timeout=15):
        import json
        #print(image_url)
        try:
            url = "http://multimedia.content-generation-qwen-vl-chat.wml.weibo.com/v2/models/content_generation_Qwen_VL_Chat/infer"
            headers = {
                'Content-Type': 'application/json',
                'Inference-Header-Content-Length': '0',
                'content-type': 'application/json'
            }
            data = {
                "request_id": "219096",
                "model_name": "content_generation_Qwen_VL_Chat",
                "input": [{"image": image_url}, {"text": prompt}]
            }
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
            #print(response.json())
            image_content = response.json()['output'][0]['output']
        except requests.exceptions.Timeout:
            #print("请求超时")
            image_content = ""
        except requests.exceptions.RequestException as e:
            #print(f"请求异常: {e}")
            image_content = ""
        return image_content
    def get_showbatch(self,mids):
        if isinstance(mids,list):
            mids_str = ','.join(mids)
        else:
            mids_str = mids
        url = 'http://i.media.api.weibo.com/2/video/show_batch.json?source=2552521548&id_type=0&part=origin,playback,statistics,playlists,author,activities,contribution,status,current_status&ids=' + mids_str
        js = self.post_api(url,method='get')
        return js
if __name__ == '__main__':
    from util import mylogging
    log = mylogging.get_logger('aa')
    A = Api_client(log)
    s = A.cid_showbatch('5178826193109138')
    print(s)
    #s = A.get_showbatch('5192325793842014')
    #print(s)
    import sys
    sys.exit()
    ss = A.get_uid_show_mul(['3211201871','5696955409'])
    print(json.dumps(ss,ensure_ascii=False))
    from jsonpath import jsonpath
    for u in ['3211201871','5696955409']:
        ins = jsonpath(ss,f"$.{u}.['idstr','created_at','text']")
        print(ins)
