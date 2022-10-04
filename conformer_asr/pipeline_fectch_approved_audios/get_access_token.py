import sys
import json
import requests

import warnings
warnings.filterwarnings("ignore")

USERNAME='khoatlv'
PASSWORD='Khoa9872134'
URL='https://115.79.194.189:9102/api/auth/login'
ACCESS_TOKEN_PATH=sys.argv[1].split('=')[-1]

data = {'username': USERNAME, 'password': PASSWORD}
headers = {'content-type': 'application/json'}

try:
    res = requests.post(url=URL, headers=headers, data=json.dumps(data), verify=False)
    if res.status_code == 200:
        res_json = json.loads(res.text)
        
        with open(ACCESS_TOKEN_PATH, 'w') as fout:
            fout.write(res_json['access'])
            
        sys.exit(0)
    
    sys.exit(-1)

except Exception as e:
    print(e)
    sys.exit(-1)


