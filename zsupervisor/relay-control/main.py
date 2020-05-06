import json
import sys
import redis
import time
import door

with open('config.json') as file:
    config = json.load(file)

server_host = config["server_host"]
port = config["port"]


def main_loop():
    print("start")
    door_service = door.Door(3)
    r = redis.Redis(host=server_host, port=port)
    pub = r.pubsub()
    pub.subscribe("open_gate")
    while True:
        data = pub.get_message()
        if data and data["data"] != 1:
            door_service.open_door()


if __name__ == "__main__":
    main_loop()
