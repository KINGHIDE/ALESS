import urllib.request
import json
import pprint
import torch
from torch import nn
import websocket
import time
import requests
import numpy as np
import random
import asyncio

RG = 300
PORT = '18080'
RANGE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
margin = 1300000

def get_token():
    obj = { 'APIPassword': apipassword }
    json_data = json.dumps(obj).encode('utf8')

    url = 'http://localhost:'+PORT+'/kabusapi/token'
    req = urllib.request.Request(url, json_data, method='POST')
    req.add_header('Content-Type', 'application/json')

    with urllib.request.urlopen(req) as res:
        print(res.status, res.reason)
        for header in res.getheaders():
            print(header)
        print()
        content = json.loads(res.read())
        return content

def get_margin():
    '''
    url = 'http://localhost:'+PORT+'/kabusapi/wallet/margin'
    req = urllib.request.Request(url, method='GET')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-API-KEY', token)

    with urllib.request.urlopen(req) as res:
        print(res.status, res.reason)
        for header in res.getheaders():
            print(header)
        print()
        content = json.loads(res.read())
        return content
    '''
    return {'MarginAccountWallet':margin}

def send_order(symbol, side, price, qty):
    '''
    if side == 1:#close
        obj = { 'Password': password,
                'Symbol': symbol,
                'Exchange': 1,
                'SecurityType': 1,
                'Side': side,
                'CashMargin': 3,
                'MarginTradeType': 3,
                'DelivType': 0,
                'AccountType': 2,
                'Qty': qty,
                'ClosePositionOrder': 0,
                'FrontOrderType': 20,
                'Price': price,
                'ExpireDay': 0
            }
    elif side == 2:#open
        obj = { 'Password': password,
                'Symbol': symbol,
                'Exchange': 1,
                'SecurityType': 1,
                'Side': side,
                'CashMargin': 2,
                'MarginTradeType': 3,
                'DelivType': 0,
                'AccountType': 2,
                'Qty': qty,
                'FrontOrderType': 20,
                'Price': price,
                'ExpireDay': 0
            }
    json_data = json.dumps(obj).encode('utf-8')

    url = 'http://localhost:'+PORT+'/kabusapi/sendorder'
    req = urllib.request.Request(url, json_data, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-API-KEY', token)

    try:
        with urllib.request.urlopen(req) as res:
            print(res.status, res.reason)
            for header in res.getheaders():
                print(header)
            print()
            content = json.loads(res.read())
            return content
    except urllib.error.HTTPError as e:
        print(e)
        content = json.loads(e.read())
        pprint.pprint(content)
    except Exception as e:
        print(e)
    '''
    id = str(random.random())
    if side == 1:      
        o_close_orders.append({'id':id, 'symbol':symbol, 'price':price, 'qty':qty-100*int(random.random())*2, 'status':bool(int(random.random())*5)})
        margin += price * o_close_orders[-1]['qty']
    else:
        o_open_orders.append({'id':id, 'symbol':symbol, 'price':price, 'qty':qty-100*int(random.random())*2, 'status':bool(int(random.random())*5)})
        margin -= price * o_close_orders[-1]['qty']
    return {'id':id}

def get_order(id):
    '''
    url = 'http://localhost:'+PORT+'/kabusapi/orders'
    params = { 'product': 0, 'id': id }

    req = urllib.request.Request('{}?{}'.format(url, urllib.parse.urlencode(params)), method='GET')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-API-KEY', token)

    try:
        with urllib.request.urlopen(req) as res:
            print(res.status, res.reason)
            for header in res.getheaders():
                print(header)
            print()
            content = json.loads(res.read())
            return content
    except urllib.error.HTTPError as e:
        print(e)
        content = json.loads(e.read())
        pprint.pprint(content)
    except Exception as e:
        print(e)
    '''
    print(open_orders+close_orders)
    for o in o_open_orders+o_close_orders:
        if o['id']==id:
            return {'Price':o['price'], 'CumQty':o['qty'], 'Symbol':o['symbol']}

def cancel_order(id):
    '''
    obj = { 'OrderID': id, 'Password': password }
    json_data = json.dumps(obj).encode('utf8')

    url = 'http://localhost:'+PORT+'/kabusapi/cancelorder'
    req = urllib.request.Request(url, json_data, method='PUT')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-API-KEY', token)

    try:
        with urllib.request.urlopen(req) as res:
            print(res.status, res.reason)
            for header in res.getheaders():
                print(header)
            print()
            content = json.loads(res.read())
            return content
    except urllib.error.HTTPError as e:
        print(e)
        content = json.loads(e.read())
        pprint.pprint(content)
    except Exception as e:
        print(e)
    '''
    return None

def judge(sequcences):
    with torch.cuda.amp.autocast():
        return softmax(model(sequcences))

def Price_Range_Group(symbol):
    time.sleep(0.15)
    url = 'http://localhost:18080/kabusapi/symbol/'+symbol+'@1'
    params = { 'addinfo': 'false' }
    req = urllib.request.Request('{}?{}'.format(url, urllib.parse.urlencode(params)), method='GET')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-API-KEY', token)

    with urllib.request.urlopen(req) as res:
        print(res.status, res.reason)
        for header in res.getheaders():
            print(header)
        print()
        content = json.loads(res.read())
        return content

def Minimum_Volality(price, PriceRangeGroup):
    if PriceRangeGroup == '10003':
        if price <= 1000:
            mv = 0.1/price*100
        elif 1000 < price <= 3000:
            mv = 0.5/price*100
        elif 3000 < price <= 5000:
            mv = 1/price*100
        elif 5000 < price <= 10000:
            mv = 1/price*100
        elif 10000 < price <= 30000:
            mv = 5/price*100
        elif 30000 < price <= 50000:
            mv = 10/price*100
        elif 50000 < price:
            mv = 10/price*100
    else:
        if price <= 1000:
            mv = 1/price*100
        elif 1000 < price <= 3000:
            mv = 1/price*100
        elif 3000 < price <= 5000:
            mv = 5/price*100
        elif 5000 < price <= 10000:
            mv = 10/price*100
        elif 10000 < price <= 30000:
            mv = 10/price*100
        elif 30000 < price <= 50000:
            mv = 50/price*100
        elif 50000 < price:
            mv = 100/price*100

    return mv

def on_message(ws, message):
    m = json.loads(message)
    s = m['Symbol']
    p = m['CurrentPrice']
    v = m['TradingVolume']
    i = symbols.index(s)
    CurrentPrices[i] = p
    CurrentVolumes[i] = v

def on_error(ws, error):
    print('--- ERROR --- ')
    print(error)

def on_close(ws):
    print('--- DISCONNECTED --- ')
    while True:
        try:
            print("Reconnecting...")
            ws.close()
            start_websocket()
            break
        except Exception as e:
            print("Reconnection failed:", e)
            time.sleep(1)  # 再接続の間隔を調整

def on_open(ws):
    print('--- CONNECTED --- ')
    
def start_websocket():
    print('connecting')
    url = 'ws://localhost:'+PORT+'/kabusapi/websocket'
    ws = websocket.WebSocketApp(url,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close,
                            on_open=on_open)
    ws.run_forever()

async def start_ws():
    loop.run_in_executor(None, start_websocket)

def start_trade():
    global last_time, open_orders, close_orders
    '''
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    header = {
        'User-Agent': user_agent
    }
    for i, s in enumerate(symbols):
        res = requests.get('https://query2.finance.yahoo.com/v8/finance/chart/'+s+'.T?interval=5m&includePrePost=true&events=div%7Csplit&range=60d', headers=header)
        quote = json.loads(res.text)['chart']['result'][0]['indicators']['quote'][0]

        indices = np.where(np.array(quote['close'])!=None)
        volumes = np.array(quote['volume'])[indices].tolist()
        prices = np.array(quote['close'])[indices].tolist()
        HistoricalPrices[i] = prices[-1000:]
        HistoricalVolumes[i] = volumes[-1000:]
        HistoricalPriceChanges[i] = [((HistoricalPrices[i][j+1]-HistoricalPrices[i][j])/HistoricalPrices[i][j]*100) for j in range(999)]
        HistoricalVolumeChanges[i] = [((HistoricalVolumes[i][j+1]-HistoricalVolumes[i][j])/HistoricalVolumes[i][j]) if HistoricalVolumes[i][j] != 0 else 0 for j in range(999)]
        HistoricalVolalities[i] = [np.mean(np.abs(HistoricalPriceChanges[i][j:j+500])) for j in range(500)]
        prg = PriceRangeGroups[i]
        HistoricalMinimumVolalities[i] = [Minimum_Volality(HistoricalPrices[i][j+1], prg) for j in range(999)]
    '''
    
    print('waiting')
    while True:
        if (time.time()-1717372800)%86400 < 10005:
            last_time = time.time()//RG*RG
            break
        else:
            time.sleep(min(1, 86400-(time.time()-1717372800)%86400))
    
    print('started')
    while True:
        if 9000<(time.time()-1717372800)%86400<12600:
            time.sleep(min(1, 12600-(time.time()-1717372800)%86400))
        elif last_time+RG < time.time():
            last_time = time.time()//RG*RG
            for i in range(len(symbols)):
                if CurrentPrices[i] != 0:
                    HistoricalPrices[i].append(CurrentPrices[i])
                    HistoricalVolumes[i].append(CurrentVolumes[i]-PreviousVolumes[i])
                    PreviousVolumes[i] = CurrentVolumes[i]
                    HistoricalPriceChanges[i].append((HistoricalPrices[i][-1]-HistoricalPrices[i][-2])/HistoricalPrices[i][-2]*100)
                    HistoricalVolalities[i].append(HistoricalVolalities[i][-1]+(abs(HistoricalPriceChanges[i][-1])-abs(HistoricalPriceChanges[i][-501])/500))
                    HistoricalMinimumVolalities[i].append(Minimum_Volality(HistoricalPrices[i][-1], PriceRangeGroups[i]))
                    HistoricalVolumeChanges[i].append((HistoricalVolumes[i][-1]-HistoricalVolumes[i][-2])/HistoricalVolumes[i][-2])

            margin = get_margin()['MarginAccountWallet']
            sequences = torch.stack([torch.tensor([i[-60:] for i in HistoricalPriceChanges], dtype=torch.float32), torch.tensor([i[-60:] for i in HistoricalVolumeChanges], dtype=torch.float32), torch.tensor([i[-60:] for i in HistoricalVolalities], dtype=torch.float32), torch.tensor([i[-60:] for i in HistoricalMinimumVolalities], dtype=torch.float32)], dim=2).to(DEVICE)
            predictions = judge(sequences)
            for pos, j, s, p in zip(positions, predictions, symbols, CurrentPrices):
                q = margin//p
                if j[1].item() > 0.52:
                    if pos[0] != 0:
                        pos[2] += 1
                    if q*p*0.99 > 1000000 and q*p*1.001 < margin:
                        open_orders.append(send_order(symbol=s, price=p, side=2, qty=q)['id'])
                        margin -= q*p*1.001

            for pos, s, p in zip(positions, symbols, CurrentPrices):
                pos[2] -= 1
                if pos[2] == 0 and pos[0] != 0:
                    close_orders.append(send_order(symbol=s, price=p, side=1, qty=pos[0])['id'])

            #for debug
            print('margin:',margin)
            print(open_orders)
            print(close_orders)
            '''
            print(HistoricalPrices[-60:])
            print(HistoricalPriceChanges[-60:])
            print(HistoricalVolumes[-60:])
            print(HistoricalVolumeChanges[-60:])
            print(HistoricalVolalities[-60:])
            print(HistoricalMinimumVolalities[-60:])
            '''
            print(HistoricalVolumes[0][-60:])
            print(HistoricalVolumeChanges[0][-60:])
            print(HistoricalPrices[0][-60:])
            print(HistoricalPriceChanges[0][-60:])
            print(HistoricalVolalities[0][-60:])
            print(HistoricalMinimumVolalities[0][-60:])
            print(predictions)
            
            time.sleep(3)

            for id in open_orders+close_orders:
                cancel_order(id)

            time.sleep(3)

            for id in open_orders:
                order = get_order(id)
                index = symbols.index(order['Symbol'])
                positions[index] = [positions[index][0]+order['CumQty'], positions[index][1]+order['CumQty']*order['Price'], RANGE]

            for id in close_orders:
                order = get_order(id)
                index = symbols.index(order['Symbol'])
                positions[index] = [positions[index][0]-order['CumQty'], positions[index][1]-order['CumQty']*order['Price'], 1]

            open_orders = []
            close_orders = []

            #for debug
            print(positions)
            with open('C:/Users/toshi/Documents/AutoTrade-main/log.txt' , 'w') as f:
                f.write('margin:'+str(margin)+'\n')
                f.write(str(open_orders)+'\n')
                f.write(str(close_orders)+'\n')
                f.write(str(predictions)+'\n')
                f.write(str(positions)+'\n')


import torch
import torch.nn as nn
import numpy as np

# LSTMモデルの定義
class StockPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, linear_drop=0.05, lstm_drop=0.0):
        super(StockPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_drop)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(linear_drop)
        self.fc7 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.Mish()

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'bn' not in name:
                nn.init.xavier_normal_(param)  # 重みの初期化
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # バイアスの初期化

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        out = self.relu(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.relu(self.bn3(self.fc3(out)))
        out = self.dropout(out)
        out = self.relu(self.bn4(self.fc4(out)))
        out = self.dropout(out)
        out = self.relu(self.bn5(self.fc5(out)))
        out = self.dropout(out)

        out = self.fc7(out)
        return out

# ハイパーパラメータの設定
input_size = 4  # 入力データの次元数
hidden_size = 4  # LSTMの隠れ層の次元数
num_layers = 1  # LSTMの層数
output_size = 2  # 出力データの次元数 (15分間の価格変化率)
# モデルのインスタンス化
model = StockPricePredictor(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('C:/Users/toshi/Documents/tmp_pth/10model80001.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
model.to(DEVICE)
model.eval()

symbols = ['6920.T', '6526.T', '8058.T', '6146.T', '8053.T', '8035.T', '7203.T', '8031.T', '4062.T', '9501.T', '7003.T', '8306.T', '6857.T', '9983.T', '8316.T', '7011.T', '6501.T', '4063.T', '8001.T', '6861.T', '6315.T', '8002.T', '6503.T', '4568.T', '9984.T', '8411.T', '6723.T', '6594.T', '9509.T', '6758.T', '7735.T', '6367.T', '9104.T', '8766.T', '6981.T', '9432.T', '2768.T', '6762.T', '6871.T', '9503.T', '6323.T', '8308.T', '2914.T', '6902.T', '9107.T', '7751.T', '4661.T', '6098.T', '8604.T', '6301.T', '4911.T', '9020.T', '3778.T', '7974.T', '9201.T', '9508.T', '7201.T', '4503.T', '3498.T', '6954.T', '1878.T', '4502.T', '5411.T', '8802.T', '6702.T', '4755.T', '9843.T', '7261.T', '9101.T', '7267.T', '6141.T', '8056.T', '9021.T', '1605.T', '8801.T', '4751.T', '6963.T', '6201.T', '9433.T', '3382.T', '4519.T', '7741.T', '6178.T', '6752.T', '1911.T', '9684.T', '5401.T', '6525.T', '8015.T', '7733.T', '8725.T', '2413.T', '9434.T', '6254.T', '6701.T', '2502.T', '4578.T', '9022.T', '5803.T', '6273.T', '5020.T', '3436.T', '6988.T', '8697.T', '7270.T', '4307.T', '2802.T', '5334.T', '7013.T', '4901.T', '8304.T', '7012.T', '4004.T', '4967.T', '7172.T', '9962.T', '9506.T', '6504.T', '5108.T', '4523.T', '3099.T', '8591.T', '9504.T', '4005.T', '5713.T', '3092.T', '8750.T', '9552.T', '7202.T', '4385.T', '4452.T', '8830.T', '9202.T', '4612.T', '5406.T', '6506.T', '7259.T', '7272.T', '6971.T', '2127.T', '6326.T', '7729.T', '7867.T', '7269.T', '1928.T', '1925.T', '8601.T', '4768.T', '2875.T', '8309.T', '6368.T', '7276.T', '1963.T', '2212.T', '9001.T', '4684.T', '5332.T', '8267.T', '7936.T', '2503.T', '4689.T', '6361.T', '5938.T', '3086.T', '4324.T', '2651.T', '5802.T', '9735.T', '9531.T', '4528.T', '5019.T', '7182.T', '4543.T', '3231.T', '6976.T', '9613.T', '8630.T', '6869.T', '3697.T', '3064.T', '6036.T', '7181.T', '7309.T', '3407.T', '8113.T', '8698.T', '6770.T', '9143.T', '8473.T', '6967.T', '1801.T', '9502.T', '9041.T', '7532.T', '7453.T', '5201.T', '3563.T', '6645.T', '8233.T', '9009.T', '4587.T', '9505.T', '1419.T', '7550.T', '7211.T', '3003.T', '2801.T', '4204.T', '3659.T', '6532.T', '6302.T', '1802.T', '7832.T', '7731.T', '4185.T', '2760.T', '4091.T', '6586.T', '6724.T', '1803.T', '3038.T', '8804.T', '6305.T', '4507.T', '4704.T', '7747.T', '9697.T', '4188.T', '8795.T', '3132.T', '3635.T', '1959.T', '2412.T', '9005.T', '4626.T', '9064.T', '9147.T', '2587.T', '2593.T', '1812.T', '6383.T', '2670.T', '3088.T', '4666.T', '4151.T', '3289.T', '9024.T', '6479.T', '9142.T', '6961.T', '4506.T', '4922.T', '7752.T', '5344.T', '6268.T', '7740.T', '9532.T', '6005.T', '7186.T', '9401.T', '3402.T', '5631.T', '7911.T', '4021.T', '6841.T', '8253.T', '8354.T', '3769.T', '6028.T', '8214.T', '3116.T', '6370.T', '9044.T', '4732.T', '6417.T', '8136.T', '7912.T', '3994.T', '5214.T', '8331.T', '4043.T', '6448.T', '6465.T', '2175.T', '9831.T', '4516.T', '9006.T', '7173.T', '7649.T', '7951.T', '3923.T', '5801.T', '6951.T', '6923.T', '3349.T', '4733.T', '6460.T', '6845.T', '9602.T', '4887.T', '6590.T', '6266.T', '1518.T', '6806.T', '8593.T', '3549.T', '2897.T', '4194.T', '4927.T', '3405.T', '2871.T', '9007.T', '6753.T', '9533.T', '9766.T', '6755.T', '5726.T', '3291.T', '6966.T', '3861.T', '2269.T', '7581.T', '7205.T', '1944.T', '7701.T', '2267.T', '1662.T', '6473.T', '2181.T', '9719.T', '5021.T', '6856.T', '9301.T', '3197.T', '4527.T', '2432.T', '6592.T', '5101.T', '7167.T', '9987.T', '6196.T', '9507.T', '8410.T', '9513.T', '2784.T', '3397.T', '3186.T', '4042.T', '8129.T', '9627.T', '8155.T', '7148.T', '6817.T', '9435.T', '8282.T', '8803.T', '5844.T', '7979.T', '8227.T', '2492.T', '4613.T', '2579.T', '9519.T', '2767.T', '5032.T', '5444.T', '4205.T', '4088.T', '4183.T', '8848.T', '4912.T', '2002.T', '9404.T', '6135.T', '3288.T', '9055.T', '5838.T', '1942.T', '3626.T', '4536.T', '6965.T', '6866.T', '6674.T', '7956.T', '9706.T', '8252.T', '4206.T', '4921.T', '5831.T', '9042.T', '6707.T', '7282.T', '5233.T', '4530.T', '4902.T', '3391.T', '6508.T', '4182.T', '6472.T', '4980.T', '3141.T', '7518.T', '7164.T', '9008.T', '5991.T', '7388.T', '5711.T', '9412.T', '4180.T', '2296.T', '6471.T', '9603.T', '2282.T', '6807.T', '2206.T', '6432.T', '5706.T', '8876.T', '6481.T', '5929.T', '8359.T', '7575.T', '5076.T', '5105.T', '6113.T', '8088.T', '4544.T', '7839.T', '3110.T', '9418.T', '9861.T', '6632.T', '7071.T', '1969.T', '4203.T', '4186.T', '4813.T', '5333.T', '8708.T', '4565.T', '2501.T', '2121.T', '1808.T', '2811.T', '1721.T', '8905.T', '1820.T', '4208.T', '8111.T', '4480.T', '9783.T', '7180.T', '5901.T', '4540.T', '6952.T', '8022.T', '2809.T', '5192.T', '5714.T', '7199.T', '2433.T', '6754.T', '2695.T', '5110.T', '6436.T', '4046.T', '4819.T', '8439.T', '2726.T', '4726.T', '2148.T', '2531.T', '5301.T', '7947.T', '7459.T', '1973.T', '2980.T', '6103.T', '9616.T', '1332.T', '3482.T', '8174.T', '6622.T', '4812.T', '6728.T', '8242.T', '9534.T', '7762.T', '6814.T', '5481.T', '2222.T', '4680.T', '4776.T', '7984.T', '3401.T', '9045.T', '7337.T', '9048.T', '4676.T', '5423.T', '8570.T', '9468.T', '3591.T', '6789.T', '7242.T', '9722.T', '4443.T', '9793.T', '8334.T', '5947.T', '9551.T', '6080.T', '2331.T', '6055.T', '8020.T', '1766.T', '2427.T', '2327.T', '9065.T', '7004.T', '4369.T', '5310.T', '8219.T', '6925.T', '6407.T', '7433.T', '7240.T', '9989.T', '4384.T', '2229.T', '4202.T', '2146.T', '2678.T', '6995.T', '8358.T', '8060.T', '3696.T', '6457.T', '2791.T', '1893.T', '8341.T', '5805.T', '8103.T', '8098.T', '5830.T', '5463.T', '3941.T', '8524.T', '7732.T', '3774.T', '3148.T', '1333.T', '6768.T', '9336.T', '7224.T', '5440.T', '3660.T', '2371.T', '4023.T', '5232.T', '8892.T', '5017.T', '2685.T', '3360.T', '1860.T', '3048.T', '2471.T', '5393.T', '2884.T', '3107.T', '9110.T', '4849.T', '7972.T', '9010.T', '3863.T', '1951.T', '8714.T', '6200.T', '7189.T', '9364.T', '9076.T', '3865.T', '7241.T', '9682.T', '1719.T', '9672.T', '8628.T', '7745.T', '1414.T', '5857.T', '2379.T', '8585.T', '3244.T', '2201.T', '9511.T', '7599.T', '8283.T', '2585.T', '9449.T', '8706.T', '4432.T', '4061.T', '8086.T', '6908.T', '9715.T', '4403.T', '7250.T', '5480.T', '3097.T', '8012.T', '8613.T', '1980.T', '8276.T', '4617.T', '9090.T', '1377.T', '4971.T', '8515.T', '5445.T', '5408.T', '4118.T', '5727.T', '3687.T', '9416.T', '3105.T', '4631.T', '8368.T', '6651.T', '2264.T', '5741.T', '6740.T', '4368.T', '8114.T', '8237.T', '6996.T', '9003.T', '2681.T', '5471.T', '8078.T', '6849.T', '5410.T', '9119.T', '8016.T', '4045.T', '4216.T', '8595.T', '3765.T', '7220.T', '4521.T', '8566.T', '8923.T', '9744.T', '6544.T', '8377.T', '4483.T', '4344.T', '2157.T', '5703.T', '8179.T', '7238.T', '1979.T', '6703.T', '6507.T', '7283.T', '6454.T', '5384.T', '6136.T', '2607.T', '6269.T', '6363.T', '7730.T', '7130.T', '6809.T', '3569.T', '7994.T', '7780.T', '7988.T', '25935.T', '4552.T', '9678.T', '7381.T', '3046.T', '6406.T', '8154.T', '4114.T', '7721.T', '9517.T', '9749.T', '7296.T', '2730.T', '8919.T', '6101.T', '9757.T', '5535.T', '7616.T', '6258.T', '6364.T', '2337.T', '9824.T', '8130.T', '7826.T', '2664.T', '8600.T', '8425.T', '7816.T', '4071.T', '7718.T', '6104.T', '4919.T', '4985.T', '9278.T', '8381.T', '6516.T', '7033.T', '3543.T', '6947.T', '9303.T', '7246.T', '4681.T', '2395.T', '7366.T', '7744.T', '1417.T', '2281.T', '1964.T', '8367.T', '7389.T', '6430.T', '6638.T', '8616.T', '4722.T', '3093.T', '167A.T', '2168.T', '8418.T', '7313.T', '4634.T', '8218.T', '6619.T', '8050.T', '6750.T', '6517.T', '2154.T', '5202.T', '1934.T', '3050.T', '9099.T', '6134.T', '3193.T', '3962.T', '7278.T', '4189.T', '1861.T', '1898.T', '4665.T', '1833.T', '9759.T', '2317.T', '7846.T', '3465.T', '6235.T', '6630.T', '8584.T', '8366.T', '6371.T', '3661.T', '6209.T', '8522.T', '4401.T', '3561.T', '1663.T', '6875.T', '9842.T', '6058.T', '8424.T', '4348.T', '6419.T', '6013.T', '6787.T', '9069.T', '1890.T', '7516.T', '9279.T', '2810.T', '2270.T', '5482.T', '3612.T', '9830.T', '9832.T', '8622.T', '7239.T', '4449.T', '4686.T', '4373.T', '1926.T', '5832.T', '3341.T', '6284.T', '3191.T', '9619.T', '8079.T', '1952.T', '7545.T', '9717.T', '7868.T', '8707.T', '6523.T', '5208.T', '2001.T', '6490.T', '2931.T', '3915.T', '7380.T', '3091.T', '7734.T', '3655.T', '9409.T', '6997.T', '8273.T', '6088.T', '3387.T', '6412.T', '9247.T', '6958.T', '2918.T', '7976.T', '7965.T', '8544.T', '7458.T', '3315.T', '2792.T', '6250.T', '6458.T', '8336.T', '4272.T', '2602.T', '2294.T', '6584.T', '9601.T', '6810.T', '3762.T', '9716.T', '7725.T', '7611.T', '3984.T', '7476.T', '5988.T', '6941.T', '8934.T', '6191.T', '9072.T', '3182.T', '7630.T', '2815.T', '7467.T', '6330.T', '7821.T', '7931.T', '6999.T', '6395.T', '8511.T', '6238.T', '7420.T', '9302.T', '8388.T', '3028.T', '9058.T', '6464.T', '4220.T', '7184.T', '4928.T', '7606.T', '4917.T', '6617.T', '6420.T', '3608.T', '7915.T', '7552.T', '2737.T', '6418.T', '5451.T', '3668.T', '9031.T', '9386.T', '8850.T', '5851.T', '2659.T', '4047.T', '9936.T', '8279.T', '9960.T', '9882.T', '4577.T', '4933.T', '5930.T', '3431.T', '3678.T', '6183.T', '5331.T', '6779.T', '7970.T', '6498.T', '4187.T', '3103.T', '1720.T', '3328.T', '3844.T', '5541.T', '4611.T', '7327.T', '4996.T', '2410.T', '3632.T', '3475.T', '7966.T', '5933.T', '8338.T', '4041.T', '7942.T', '7613.T', '3880.T', '5269.T', '5352.T', '2733.T', '2326.T', '2301.T', '8609.T', '6804.T', '3593.T', '1813.T', '3087.T', '4345.T', '7715.T', '2489.T', '8011.T', '4053.T', '8346.T', '8133.T', '9746.T', '5191.T', '3679.T', '6278.T', '3458.T', '9216.T', '6788.T', '6653.T', '5602.T', '2910.T', '7280.T', '2120.T', '4028.T', '1961.T', '6455.T', '7354.T', '3167.T', '6222.T', '2440.T', '4553.T', '9600.T', '9948.T', '1941.T', '7944.T', '1515.T', '8091.T', '8051.T', '9424.T', '1835.T', '4044.T', '6727.T', '9742.T', '9267.T', '4633.T', '7995.T', '7383.T', '8361.T', '2462.T', '3445.T', '5288.T', '7226.T', '9974.T', '1968.T', '6376.T', '4547.T', '7888.T', '9934.T', '5302.T', '6287.T', '8153.T', '3101.T', '7817.T', '3656.T', '5852.T', '2292.T', '9075.T', '9692.T', '9605.T', '4974.T', '4714.T', '4559.T', '4956.T']
symbols = symbols[:20]

apipassword = '9BjmVz5OZF'
password = '9BjmVz5OZF'
token = get_token()['Token']
model = model
softmax = nn.Softmax(dim=1)

symbols = [s[:-2] for s in symbols]
CurrentPrices = [0.0 for i in symbols]
CurrentVolumes = [0.0 for i in symbols]
PreviousVolumes = [0.0 for i in symbols]
HistoricalPrices = [[] for i in symbols]
HistoricalPriceChanges = [[] for i in symbols]
HistoricalVolumes = [[] for i in symbols]
HistoricalVolumeChanges = [[] for i in symbols]
HistoricalVolalities = [[] for i in symbols]
HistoricalMinimumVolalities = [[] for i in symbols]
PriceRangeGroups = [Price_Range_Group(s)['PriceRangeGroup'] for s in symbols]

#
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
header = {
    'User-Agent': user_agent
}
for i, s in enumerate(symbols):
    res = requests.get('https://query2.finance.yahoo.com/v8/finance/chart/'+s+'.T?interval=5m&includePrePost=true&events=div%7Csplit&range=60d', headers=header)
    quote = json.loads(res.text)['chart']['result'][0]['indicators']['quote'][0]

    indices = np.where(np.array(quote['close'])!=None)
    volumes = np.array(quote['volume'])[indices].tolist()
    prices = np.array(quote['close'])[indices].tolist()
    HistoricalPrices[i] = prices[-1000:]
    HistoricalVolumes[i] = volumes[-1000:]
    HistoricalPriceChanges[i] = [((HistoricalPrices[i][j+1]-HistoricalPrices[i][j])/HistoricalPrices[i][j]*100) for j in range(999)]
    HistoricalVolumeChanges[i] = [((HistoricalVolumes[i][j+1]-HistoricalVolumes[i][j])/HistoricalVolumes[i][j]) if HistoricalVolumes[i][j] != 0 else 0 for j in range(999)]
    HistoricalVolalities[i] = [np.mean(np.abs(HistoricalPriceChanges[i][j:j+500])) for j in range(500)]
    prg = PriceRangeGroups[i]
    HistoricalMinimumVolalities[i] = [Minimum_Volality(HistoricalPrices[i][j+1], prg) for j in range(999)]
#
url = 'http://localhost:'+PORT+'/kabusapi/unregister/all'
req = urllib.request.Request(url, method='PUT')
req.add_header('Content-Type', 'application/json')
req.add_header('X-API-KEY', token)

with urllib.request.urlopen(req) as res:
    print(res.status, res.reason)
    for header in res.getheaders():
        print(header)
    print()
    content = json.loads(res.read())
    pprint.pprint(content)

obj = {'Symbols':[{'Symbol':s, 'Exchange':1} for s in symbols]}
json_data = json.dumps(obj).encode('utf8')
url = 'http://localhost:'+PORT+'/kabusapi/register'
req = urllib.request.Request(url, json_data, method='PUT')
req.add_header('Content-Type', 'application/json')
req.add_header('X-API-KEY', token)

with urllib.request.urlopen(req) as res:
    print(res.status, res.reason)
    for header in res.getheaders():
        print(header)
    print()
    content = json.loads(res.read())
    pprint.pprint(content)

open_orders = []
close_orders = []
o_open_orders = []
o_close_orders = []
positions = [[0, 0, 0] for i in symbols]
last_time = time.time()//RG*RG

wt = 0
lt = 0
wa = 0
la = 0

asyncio.set_event_loop(asyncio.new_event_loop())
loop = asyncio.get_event_loop()

asyncio.run(start_ws())
start_trade()
