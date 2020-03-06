import requests

data = {
    "strIPCamera": "192.168.1.150",
    "strAddress": "192.168.1.150",
    "strNetMask": "255.255.255.0",
    "strGateWay": "192.168.1.2",
    "strIPSrv": "192.168.1.23",
    "strLane": "363"
}

response = requests.post("http://localhost:8080/facedetect/ipconfig", json=data)
print(response.text)
