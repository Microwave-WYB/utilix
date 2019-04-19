import os
import requests
import json
import datetime
import logging
from utilix.config import Config

#Config the logger:
logger = logging.getLogger("utilix")
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

#Config Utilix
config = Config()
PREFIX = config.get('RunDB', 'rundb_api_url')
BASE_HEADERS = {'Content-Type': "application/json", 'Cache-Control': "no-cache"}

def Responder(func):

    def LookUp():
        return_dict = {
            #taken from https://github.com/kennethreitz/requests/blob/master/requests/status_codes.py
            # Informational.
            100: ('continue',),
            101: ('switching_protocols',),
            102: ('processing',),
            103: ('checkpoint',),
            122: ('uri_too_long', 'request_uri_too_long'),
            200: ('ok', 'okay', 'all_ok', 'all_okay', 'all_good', '\\o/', '✓'),
            201: ('created',),
            202: ('accepted',),
            203: ('non_authoritative_info', 'non_authoritative_information'),
            204: ('no_content',),
            205: ('reset_content', 'reset'),
            206: ('partial_content', 'partial'),
            207: ('multi_status', 'multiple_status', 'multi_stati', 'multiple_stati'),
            208: ('already_reported',),
            226: ('im_used',),

            # Redirection.
            300: ('multiple_choices',),
            301: ('moved_permanently', 'moved', '\\o-'),
            302: ('found',),
            303: ('see_other', 'other'),
            304: ('not_modified',),
            305: ('use_proxy',),
            306: ('switch_proxy',),
            307: ('temporary_redirect', 'temporary_moved', 'temporary'),
            308: ('permanent_redirect',
                  'resume_incomplete', 'resume',),  # These 2 to be removed in 3.0

            # Client Error.
            400: ('bad_request', 'bad'),
            401: ('unauthorized',),
            402: ('payment_required', 'payment'),
            403: ('forbidden',),
            404: ('not_found', '-o-'),
            405: ('method_not_allowed', 'not_allowed'),
            406: ('not_acceptable',),
            407: ('proxy_authentication_required', 'proxy_auth', 'proxy_authentication'),
            408: ('request_timeout', 'timeout'),
            409: ('conflict',),
            410: ('gone',),
            411: ('length_required',),
            412: ('precondition_failed', 'precondition'),
            413: ('request_entity_too_large',),
            414: ('request_uri_too_large',),
            415: ('unsupported_media_type', 'unsupported_media', 'media_type'),
            416: ('requested_range_not_satisfiable', 'requested_range', 'range_not_satisfiable'),
            417: ('expectation_failed',),
            418: ('im_a_teapot', 'teapot', 'i_am_a_teapot'),
            421: ('misdirected_request',),
            422: ('unprocessable_entity', 'unprocessable'),
            423: ('locked',),
            424: ('failed_dependency', 'dependency'),
            425: ('unordered_collection', 'unordered'),
            426: ('upgrade_required', 'upgrade'),
            428: ('precondition_required', 'precondition'),
            429: ('too_many_requests', 'too_many'),
            431: ('header_fields_too_large', 'fields_too_large'),
            444: ('no_response', 'none'),
            449: ('retry_with', 'retry'),
            450: ('blocked_by_windows_parental_controls', 'parental_controls'),
            451: ('unavailable_for_legal_reasons', 'legal_reasons'),
            499: ('client_closed_request',),

            # Server Error.
            500: ('internal_server_error', 'server_error', '/o\\', '✗'),
            501: ('not_implemented',),
            502: ('bad_gateway',),
            503: ('service_unavailable', 'unavailable'),
            504: ('gateway_timeout',),
            505: ('http_version_not_supported', 'http_version'),
            506: ('variant_also_negotiates',),
            507: ('insufficient_storage',),
            509: ('bandwidth_limit_exceeded', 'bandwidth'),
            510: ('not_extended',),
            511: ('network_authentication_required', 'network_auth', 'network_authentication'),
            }
        return return_dict

    def func_wrapper(*args, **kwargs):
        st = func(*args, **kwargs)
        if st.status_code != 200:
            logger.error("HTTP(s) request says: {0} (Code {1})".format(LookUp()[st.status_code][0], st.status_code))
        return st
    return func_wrapper

class Token:
    """
    Object handling tokens for runDB API access.
    
    """
    def __init__(self, path=os.path.join(os.environ['HOME'], ".dbtoken")):
        # if token path exists, read it in. Otherwise make a new one
        if os.path.exists(path):
            with open(path) as f:
                json_in = json.load(f)
                self.string = json_in['string']
                self.creation_time = json_in['creation_time']
        else:
            self.string = self.new_token()
            self.creation_time = datetime.datetime.now().timestamp()
        self.path = path

        # for writing to disk
        self.json = dict(string=self.string, creation_time=self.creation_time)
        # save the token json to disk
        self.write()
        # refresh if needed
        self.refresh()

    def __call__(self):
        return self.string

    def new_token(self):
        path = PREFIX + "/login"
        data=json.dumps({"username": config.get('RunDB', 'rundb_api_user'),
                         "password": config.get('RunDB', 'rundb_api_password')})
        response = requests.post(path, data=data, headers=BASE_HEADERS)
        return json.loads(response.text)['access_token']

    @property
    def is_valid(self):
        # TODO do an API call for this instead?
        return datetime.datetime.now().timestamp() - self.creation_time < 24*60*60

    def refresh(self):
        # if valid, don't do anything
        if self.is_valid:
            logger.debug("Token is valid")
            return
        # update the token string
        url = PREFIX + "/refresh"
        headers = BASE_HEADERS.copy()
        headers['Authorization'] = "Bearer {string}".format(string=self.string)
        response = requests.get(url, headers=headers)
        # if rewew fails, try logging back in
        if response.status_code is None or response.status_code is not 200:
            self.string = self.new_token()
            self.creation_time = datetime.datetime.now().timestamp()
        else:
            self.string = json.loads(response.text)['access_token']
        # write out again
        self.write()
        logger.debug("Token refreshed")

    def write(self):
        with open(self.path, "w") as f:
            json.dump(self.json, f)


class DB():
    """Wrapper around the RunDB API"""

    def __init__(self, token_path=os.path.join(os.environ['HOME'], ".dbtoken")):
        # Takes a path to serialized token object
        token = Token(token_path)

        self.headers = BASE_HEADERS.copy()
        self.headers['Authorization'] = "Bearer {token}".format(token=token())

    #Helper:
    @Responder
    def _get(self, url):
        return requests.get(PREFIX + url, headers=self.headers)

    @Responder
    def _put(self, url, data):
        return requests.put(PREFIX + url, data, headers=self.headers)

    @Responder
    def _post(self, url, data):
        return requests.post(PREFIX + url, data, headers=self.headers)

    @Responder
    def _delete(self, url, data):
        return requests.delete(PREFIX + url, data=data, headers=self.headers)

    def get_name(self, number, detector='tpc'):
        # TODO check against the detector, if necessary
        url = "/runs/number/{number}/filter/detector".format(number=number)
        response = json.loads(self._get(url).text)
        return response['results']['name']

    def get_number(self, name, detector='tpc'):
        url = "/runs/name/{name}/filter/detector".format(name=name)
        response = json.loads(self._get(url).text)
        return response['results']['number']

    def get_doc_by_number(self, number):
        # return the whole run doc for this run number
        url = '/runs/number/{num}'.format(num=number)
        return json.loads(self._get(url).text)['results']

    def get_doc_by_name(self, name):
        # return the whole run doc for this run number
        url = '/runs/name/{num}'.format(num=name)
        return json.loads(self._get(url).text)['results']

    def get_data_by_number(self, number):
        # return the whole run doc for this run number
        url = '/runs/number/{num}/data'.format(num=number)
        return json.loads(self._get(url).text)['results']

    def get_data_by_name(self, name):
        # return the whole run doc for this run number
        url = '/runs/name/{num}/data'.format(num=name)
        return json.loads(self._get(url).text)['results']

    def update_data_by_number(self, number, datum):
        datum = json.dumps(datum)
        url = '/run/number/{num}/data/'.format(num=number)
        return self._post(url, data=datum)

    def update_data_by_name(self, name, datum):
        datum = json.dumps(datum)
        url = '/run/name/{num}/data/'.format(num=name)
        return self._post(url, data=datum)

    def delete_data_by_number(self, number, datum):
        datum = json.dumps(datum)
        url = '/run/number/{num}/data/'.format(num=number)
        return self._delete(url, data=datum)

    def delete_data_by_name(self, name, datum):
        datum = json.dumps(datum)
        url = '/run/name/{num}/data/'.format(num=name)
        return self._delete(url, data=datum)

    def query(self, page_num):
        url = '/runs/page/{page_num}'.format(page_num=page_num)
        response = json.loads(self._get(url).text)
        return response['results']
    
    def query_by_source(self, source, page_num):
        url = '/runs/source/{source}/page/{page_num}'.format(source=source, page_num=page_num)
        response = json.loads(self._get(url).text)
        return response['results']

    def query_by_tag(self, tag, page_num):
        url = '/runs/tag/{tag}/page/{page_num}'.format(tag=tag, page_num=page_num)
        response = json.loads(self._get(url).text)
        return response['results']


# for testing
def test():
    db = DB()
    print(db.get_doc_by_number(2000))


if __name__ == "__main__":
    test()
