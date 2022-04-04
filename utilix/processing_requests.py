import os
from typing import Literal
import rframe
import uuid
import datetime
import getpass
import pydantic
import requests
from warnings import warn
from . import uconfig

CACHE = {}
DEFAULT_ENV = '2022.03.5'
ENV_TAGS_URL = 'https://api.github.com/repos/xenonnt/base_environment/git/matching-refs/tags/'

API_URL = 'https://api.xedocs.yossisprojects.com'

if uconfig is not None:
    API_URL = uconfig.get('cmt2', 'api_url', fallback=API_URL)


RSE_TYPE = Literal['SURFSARA_USERDISK',
              'SDSC_USERDISK', 
              'LNGS_USERDISK', 
              'UC_OSG_USERDISK', 
              'UC_DALI_USERDISK', 
              'CNAF_USERDISK']


def xeauth_user():
    return uconfig.get('cmt2', 'api_user', fallback='unknown')


def get_envs():
    r = requests.get(ENV_TAGS_URL)
    if not r.ok:
        return []
    tags = r.json()
    tagnames = [tag['ref'][10:] for tag in tags if tag['ref'] ]
    return tagnames

def default_env():
    cond_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if cond_env.startswith('XENONnT_'):
        return cond_env.replace('XENONnT_', '')
    envs = get_envs()
    if envs:
        return envs[-1]
    return DEFAULT_ENV

class ProcessingRequest(rframe.BaseSchema):
    '''Schema definition for a processing request
    '''
    _ALIAS = 'processing_requests'

    data_type: str = rframe.Index()
    lineage_hash: str = rframe.Index()
    run_id: str = rframe.Index()
    destination: RSE_TYPE = rframe.Index(default='UC_DALI_USERDISK')
    user: str = pydantic.Field(default_factory=xeauth_user)
    request_date: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.utcnow)
    
    priority: int = -1

    comments: str = ''
    
    
    def pre_update(self, datasource, new):
        if new.user != self.user:
            raise ValueError(new.user)
        if new.run_id != self.run_id:
            raise ValueError(new.run)

    @classmethod
    def default_datasource(cls):
        return processing_api()

    def latest_context(self):
        import utilix
        import pymongo

        contexts = utilix.xent_collection('contexts')
        ctx = contexts.find_one({f'hashes.{self.data_type}': self.lineage_hash},
                  projection={'name': 1, 'env': '$tag', '_id': 0},
                  sort=[('date_added', pymongo.DESCENDING)])
        return ctx

    def create_job(self):
        kwargs = self.latest_context()
        kwargs.update(self.dict())
        return ProcessingJob(**kwargs)


class ProcessingJob(rframe.BaseSchema):
    _ALIAS = 'processing_jobs'

    job_id: uuid.UUID = rframe.Index(default_factory=uuid.uuid4)
    destination: RSE_TYPE = rframe.Index()
    env: str = rframe.Index()
    context: str = rframe.Index()
    data_type: str = rframe.Index()
    run_id: str = rframe.Index()
    lineage_hash: str = rframe.Index()
    
    location: RSE_TYPE = None
    submission_time: datetime.datetime = None
    completed: bool = False
    progress: int = 0
    error: str = ''

    def create_workflow(self):
        raise NotImplementedError

    def submit(self):
        raise NotImplementedError


def xeauth_login(readonly=True):
    try:
        import xeauth
        scopes = ['read:all'] if readonly else ['read:all', 'write:all']
        audience = uconfig.get('cmt2', 'api_audience', fallback='https://api.cmt.xenonnt.org')

        username = uconfig.get('cmt2', 'api_user', fallback='unknown')
            
        password = uconfig.get('cmt2', 'api_password', fallback=None)

        if password is None:
            xetoken = xeauth.login(scopes=scopes, audience=audience)
        else:
            xetoken = xeauth.user_login(username,
                                        password,
                                        scopes=scopes)

        uconfig.set('cmt2', 'api_user', xetoken.username)
        
        return xetoken.access_token
    except: 
        return None


def valid_token(token, readonly=True):
    if readonly:
        scope = 'read:all'
    else:
        scope = 'write:all'

    try:
        import xeauth
        claims = xeauth.certs.extract_verified_claims(token)
        assert scope in claims.get('scope', '')
    except:
        return False

    return True

def processing_api(token=None, readonly=False):

    if token is None:
        token = uconfig.get('cmt2', 'api_token', fallback=None)

    if not valid_token(token, readonly=readonly):
        token = None

    if token is None:
        token = xeauth_login(readonly=readonly)
    
    if token is None:
        token = getpass.getpass('API token: ')
    
    headers = {}
    if token:
        headers['Authorization'] = f"Bearer {token}"
        token = uconfig.set('cmt2', 'api_token', token)

    client = rframe.RestClient(f'{API_URL}/processing_requests',
                                 headers=headers,)
    return client

try:
    import strax
    import tqdm

    @strax.Context.add_method
    def request_processing(context, run_ids, data_type,
                           priority=-1, comments='',
                           destination='UC_DALI_USERDISK',
                           token=None, submit=True):
        client = processing_api(token=token, readonly=False)

        run_ids = strax.to_str_tuple(run_ids)
        requests = []
        for run_id in tqdm.tqdm(run_ids, desc='Requesting processing'):
            
            lineage_hash = context.key_for(run_id, data_type).lineage_hash

            kwargs = dict(data_type=data_type, 
                          lineage_hash=lineage_hash,
                          run_id=run_id,
                          priority=priority,
                          destination=destination,
                          comments=comments)
            
            request = ProcessingRequest(**kwargs)
            requests.append(request)
            if submit:
                request.save(client)

        if len(requests) == 1:
            return requests[0]
        return requests

except ImportError:
    pass