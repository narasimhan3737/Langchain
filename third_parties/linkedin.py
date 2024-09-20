import os
import requests
from dotenv import load_dotenv
import json 
load_dotenv()



def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False ):
    with open('./third_parties/linkedin_profile.json') as f:
        d = json.load(f)
        print(d)

        return d


        """ if mock:
            linkedin_profile_url = ""
            response = requests.get(linkedin_profile_url,timeout=10)
        else:
            api_endpoint = ""
            header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
            response = requests.get(
                api_endpoint,
                params="url": linkedin_profile_url,
                headers=header_dic,
                timeout=10,
            )
        
        data = response.json()

        return data """

if __name__ == "__main__":
    print(scrape_linkedin_profile(linkedin_profile_url=""))
