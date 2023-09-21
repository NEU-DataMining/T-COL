import SparkApi
import argparse
import pandas as pd
import time

def getText(role, content):
    text = []
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def generate(sample,cols,target,cl):
    appid = "e4bf5f95"
    domain = "generalv2"
    api_secret = "MjNjYWMzYWVkNzA0MDhkMzA4OGY4OWFm"
    api_key = "732413627c355be1caa8a625b9cb7727"
    Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"
    prompt = "这是一个样例:%s\n它的值分别表示:%s\n它的类别是%s，表示%s，请为它生成5个反事实解释样例，并以字典的形式给出反事实样例中每个属性的特征值"%(sample,cols,target,cl)
    question = getText("user", prompt)
    SparkApi.answer = ""
    SparkApi.main(appid,api_key,api_secret,Spark_url,domain,question)
    return SparkApi.answer

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--data", choices=["german", "adult", "water", "titanic", "airline", "phoneme"], default="adult", help="Choosing a dataset.")
    args = parse.parse_args()
    if args.data == "german":
        c = "credits"
        root_path = "German/"
        pass
    elif args.data == "adult":
        c = "income"
        root_path = "Adult/"
        pass
    elif args.data == "water":
        c = "Potability"
        root_path = "Water/"
        pass
    elif args.data == "airline":
        c = "Flight Status"
        root_path = "Airline/"
        pass
    elif args.data == "phoneme":
        c = "class"
        root_path = "Phoneme/"
        pass
    elif args.data == "titanic":
        c = "Survived"
        root_path = "Titanic/"
        pass
    CEs = []
    samples = pd.read_csv(root_path+"samples.csv",header=0)
    samples.astype(str)
    cols = ",".join(samples.columns.tolist()[:-1])
    for sample in samples.iterrows():
        t = sample[1][c]
        sample[1].drop(c, inplace=True)
        s = ",".join(sample[1].astype(str).tolist())
        CE = generate(s,cols,t,c)
        CEs.append(CE)
        time.sleep(3)
        pass

    with open(root_path+"LLM.txt",'w') as f:
        f.write("\n".join(map(str,CEs)))
        pass
    f.close()