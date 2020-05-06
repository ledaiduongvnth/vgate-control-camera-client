from flask import Flask
from flask import request
from flask import Response
import os
import socket

app = Flask(__name__)


def check_data(data):
    try:
        strIPCamera = data["strIPCamera"]
        strAddress = data["strAddress"]
        strNetMask = data["strNetMask"]
        strGateWay = data["strGateWay"]
        strIPSrv = data["strIPSrv"]
        strLane = data["strLane"]
    except:
        return False

    try:
        socket.inet_aton(strIPCamera)
        socket.inet_aton(strAddress)
        socket.inet_aton(strNetMask)
        socket.inet_aton(strGateWay)
        socket.inet_aton(strIPSrv)
        assert 0 < int(strLane) < 100
    except:
        return False

    return True


@app.route('/facedetect/ipconfig', methods=['POST'])
def config():
    data = request.json
    print(request.json)
    ok = check_data(data)
    if ok:
        strIPCamera = data["strIPCamera"]
        strAddress = data["strAddress"]
        strNetMask = data["strNetMask"]
        strGateWay = data["strGateWay"]
        strIPSrv = data["strIPSrv"]
        strLane = data["strLane"]
        ip_config_data = "\nauto eth0" \
                         "\niface eth0 inet static" + \
                         "\naddress " + strAddress + \
                         "\nnetmask " + strNetMask + \
                         "\ngateway " + strGateWay
        with open('/etc/network/interfaces.d/eth0', 'w') as file:
            file.write(ip_config_data)

        camera_cliet_config_file = "../../config/config.json"

        with open(camera_cliet_config_file, 'r') as file:
            # read a list of lines into data
            camera_client_config_data = file.readlines()

        camera_client_config_data[1] = '  "strLane": "' + str(strLane) + '",\n'
        camera_client_config_data[2] = '  "multiple-camera-host": "' + strIPSrv + '",\n'
        camera_client_config_data[3] = '  "camera_source": "' + strIPSrv + '",\n'


        with open(camera_cliet_config_file, 'w') as file:
            file.writelines(camera_client_config_data)


        # os.system('reboot now')
        return Response("{'intResultCode':'200', 'strDescription':'successful'}", status=200,
                        mimetype='application/json')

    else:
        return Response("{'intResultCode':'201', 'strDescription':'unsuccessful'}", status=201,
                        mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
