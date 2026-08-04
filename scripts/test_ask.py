import requests, json

url = 'http://127.0.0.1:8000/ask'
payload = {'text': 'Where are the fuses located?'}
try:
    r = requests.post(url, json=payload, timeout=30)
    print(r.status_code)
    # save response to file for inspection (safer for PowerShell capture)
    try:
        with open('scripts/last_ask_response.json', 'w', encoding='utf-8') as f:
            json.dump(r.json(), f, indent=2)
        print('saved to scripts/last_ask_response.json')
    except Exception:
        with open('scripts/last_ask_response.txt', 'w', encoding='utf-8') as f:
            f.write(r.text)
        print('saved to scripts/last_ask_response.txt')
except Exception as e:
    print('request error:', e)
