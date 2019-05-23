import requests
import json
from data_handler import *



class SessionDataService:
    DSI_URL_BASE = "https://wc-dsi-api-prod.azurewebsites.net/sessions"
    
    def __init__(self, store_id):
        self.storeId = store_id

    def get_sessions_meta(self, start_date, end_date, count, to_skip = 0, paging=25):
        url = SessionDataService.DSI_URL_BASE + "?start=" + start_date + "&end=" + end_date + "&storeId="+self.storeId+"&customerId=&type=cart&skip={}&limit={}"

        sessions = []
        count += to_skip
        for skip in range(to_skip, skip+count, paging):
            limit = min(count, paging)
            response = requests.get(url.format(skip, limit))
            parsed = json.loads(response.content)
            sessions += parsed["data"]
        
        return sessions

    def get_session_content(self, session_id):
        url = SessionDataService.DSI_URL_BASE + "/sessions/" + self.storeId + "/" + session_id
        response = requests.get(url)
        return json.loads(response.content)

    def get_sessions(self, start_date, end_date, count, to_skip = 0):
        sessions_meta = self.get_sessions_meta(start_date, end_date, count, to_skip)
        session_handlers = []
        for s in sessions_meta:
            session_handlers.append(SessionHandler(self.get_session_content(s["id"]))

        return session_handlers
        





    



