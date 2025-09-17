# -*- coding: utf-8 -*-
"""
@Author: jiangtao16
@Date: 2024/6/24
@Description: 
"""
# coding:utf-8

import codecs
import hmac
import traceback
from hashlib import md5, sha1
from http.client import responses
from urllib.parse import urlencode, urlparse, parse_qs, ParseResult

import requests
import time
from requests.auth import AuthBase


class KidAuth(AuthBase):
    def __init__(self, kid, passwd):
        self.kid = kid
        self.passwd = passwd
        self.expired = 360

    def __call__(self, r):
        content_md5 = ''
        if r.method == "POST" or r.method == "PUT":
            content_md5 = md5(r.body.encode(encoding='UTF-8')).hexdigest()
            r.headers['Content-MD5'] = content_md5
        content_type = r.headers.get('Content-Type', '')
        expired = str(time.time() + self.expired)
        r.headers['Expires'] = expired
        stringtosign = '\n'.join([r.method,
                                  content_md5,
                                  content_type,
                                  expired,
                                  r.path_url])
        signature = hmac.new(self.passwd.encode("utf-8"), stringtosign.encode("utf-8"), sha1).digest()
        ssig = codecs.encode(signature, 'base64')[5:15]
        r.headers['Authorization'] = 'sinawatch %s:%s' % (self.kid, ssig.decode())
        return self


class HTTPError(Exception):
    def __init__(self, status_code, log_message=None):
        self.status_code = status_code
        self.log_message = log_message

    def __str__(self):
        message = "HTTP %d: %s" % (
            self.status_code, responses.get(self.status_code, 'Unknown'))
        if self.log_message:
            return message + " (" + self.log_message + ")"
        else:
            return message


class ApiError(HTTPError):
    def __init__(self, status_code, **kwargs):
        super(ApiError, self).__init__(status_code, kwargs.get('error', kwargs.get('message')))
        self.request_id = kwargs.get('request-id')
        self.error = kwargs.get('error')


class ApiClient(object):
    def __init__(self, host, kid, passwd):
        self.auth = KidAuth(kid, passwd)
        self.url = urlparse(host)
        self._query = parse_qs(self.url.query)

    def post(self, path, query=None, **kwargs):
        url = self.geturl(path, query)
        return self.response(requests.post(url, auth=self.auth, **kwargs))

    def get(self, path, query=None, **kwargs):
        url = self.geturl(path, query)
        return self.response(requests.get(url, auth=self.auth, **kwargs))

    def delete(self, path, query=None, **kwargs):
        url = self.geturl(path, query)
        return self.response(requests.delete(url, auth=self.auth, **kwargs))

    def response(self, resp):
        if resp.status_code >= 400:
            if resp.headers.get('Content-Type') == "application/json":
                raise ApiError(resp.status_code, **resp.json())
            raise HTTPError(resp.status_code, resp.content)
        if resp.status_code == 201 or resp.status_code == 204:
            return None
        data = resp.json()
        return data

    def geturl(self, path, query):
        if not query:
            query = {}
        if path.startswith('/'):
            path = path[1:]
        return ParseResult(self.url.scheme, self.url.netloc, path, self.url.params,
                           urlencode(query, doseq=True),
                           self.url.fragment).geturl()


if __name__ == '__main__':
    kid='2019052715'
    passwd='17cc26DOfNTnk9bZlgwNvoiKk39T8Z'
    ALERT_KID = kid
    ALERT_PASSWORD = passwd
    ALERT_HOST = 'http://iconnect.monitor.sina.com.cn'
    api = ApiClient(ALERT_HOST, ALERT_KID, ALERT_PASSWORD)
    try:
        res = api.post("/v1/alert/send",timeout=3,
                 data={
                     'object': '',
                     'service': "service",
                     #'mailto': "lingjie3,chuanyun",
                     'dingto': "jiangtao16",
                     'wechatto': "jiangtao16",
                     'sv': '新浪内容加速平台',
                     'mailfrom': '新浪内容加速平台',#邮件发送名称
                     'subject': "告警主题",
                     'content': "邮件内容",
                     'html': "1"})
        print(res)
    except Exception as e:
        traceback.print_stack()

