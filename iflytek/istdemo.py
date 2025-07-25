import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识
result = ''


# 收到websocket消息的处理
def on_message(ws, message):
    global result
    try:
        code = json.loads(message)["header"]["code"]
        sid = json.loads(message)["header"]["sid"]
        if code != 0:
            errMsg = json.loads(message)
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:

            print(1, json.loads(message))
            # data = json.loads(message)["data"]["result"]["ws"]
            pgs = json.loads(message)["payload"]["result"]["pgs"]
            ws_list = json.loads(message)['payload']['result']['ws']
            status = json.loads(message)["header"]["status"]
            if pgs == 'apd':
                for i in ws_list:
                    for w in i["cw"]:
                        result += w["w"]

            if status == 2:
                ws.close()

            # print("sid:%s call success!,data is:%s" % (sid, json.dumps(data, ensure_ascii=False)))
    except Exception as e:
        print("receive msg,but parse exception:", e)


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws, close, close_2):
    print("### closed ###", close, close_2)


# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        frameSize = 1280  # 每一帧的音频大小
        intervel = 0.04  # 发送音频间隔(单位:s)
        status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧

        with open(r".\zhongwen.wav", "rb") as fp:
            while True:
                buf = fp.read(frameSize)
                # 文件结束
                if not buf:
                    status = STATUS_LAST_FRAME
                # 第一帧处理
                # 发送第一帧音频，带business 参数
                # appid 必须带上，只需第一帧发送
                if status == STATUS_FIRST_FRAME:

                    # d = {"common": "",
                    #      "business": {"domain": "iat", "language": "en", "accent": "mandarin", "eos": 1000, "aue": "raw"},
                    #      "data": {"status": 0, "format": "audio/L16;rate=16000",
                    #               "audio": str(base64.b64encode(buf), 'utf-8'),
                    #               "encoding": "raw"}}
                    d = {"header": {
                        "traceId": "traceId123456",
                        "appId": "123456",
                        "bizId": "39769795890",
                        "status": 0
                        },

                        "payload": {
                            "audio": {
                                "audio": str(base64.b64encode(buf), 'utf-8'),
                                # "audio": str(base64.b64encode(buf), 'utf-8'),
                            }
                        }}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                # 中间帧处理
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"header": {
                        "traceId": "traceId123456",
                        "appId": "123456",
                        "bizId": "39769795890",
                        "status": 1
                    },
                        "payload": {
                            "audio": {
                                "audio": str(base64.b64encode(buf), 'utf-8'),
                                # "audio": str(base64.b64encode(buf), 'utf-8'),
                            }
                        }}
                    ws.send(json.dumps(d))
                # 最后一帧处理
                elif status == STATUS_LAST_FRAME:
                    d = {"header": {
                        "traceId": "traceId123456",
                        "appId": "123456",
                        "bizId": "39769795890",
                        "status": 2
                    },
                        "payload": {
                            "audio": {
                                "audio": str(base64.b64encode(buf), 'utf-8'),
                                # "audio": str(base64.b64encode(buf), 'utf-8'),
                            }
                        }}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                # 模拟音频采样间隔
                time.sleep(intervel)
        # ws.close()

    thread.start_new_thread(run, ())


if __name__ == "__main__":
    #for i in range(0, 1):
    # 测试时候在此处正确填写相关信息即可运行
    time1 = datetime.now()
    wsUrl = "ws://94.74.125.108:9990/tuling/ast/v3"

    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    print(ws.on_open)
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    print(2, result)
    time2 = datetime.now()
    print(time2 - time1)
    