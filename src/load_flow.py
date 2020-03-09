import os
import urllib.request

if not os.path.exists('./models/VAE_NAF_flow.pt'):
    print('Downloading model... \n')
    with open("./model_url.txt") as file:
        url = file.read()
        urllib.request.urlretrieve(url, './models/VAE_NAF_flow.pt')
os.remove("./model_url.txt")
print('Success! \n')