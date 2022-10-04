import sys
import json
import requests
import pickle

import warnings
warnings.filterwarnings("ignore")


USERNAME='khoatlv'
ACCESS_TOKEN_PATH=sys.argv[1].split('=')[-1]
SENTENCE_DICT_PATH=sys.argv[2].split('=')[-1]
URL='https://115.79.194.189:9102/api/admin/sentence'

with open(ACCESS_TOKEN_PATH, 'r') as token:
    ACCESS_TOKEN = token.readline()

params = {'limit': -1, 'offset': 0}
headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + ACCESS_TOKEN.strip()}

try:
    res = requests.get(url=URL, headers=headers, params=json.dumps(params), verify=False)
    if res.status_code == 200:
        sentence_dict = dict()
        res_json = json.loads(res.text)
        sentences = res_json['data']['sentences']
        total_sentences = res_json['data']['total_sentences']
        sentence_dict = {int(sentence['id']): sentence['sentence'] for sentence in sentences}
        
        with open(SENTENCE_DICT_PATH, 'wb') as fout:
            pickle.dump(sentence_dict, fout)
            
        exit(0)
    exit(-1)

except Exception as e:
    print(e)
    exit(-1)




