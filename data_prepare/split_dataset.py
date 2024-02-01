import os
import random
import math

os.chdir(os.path.dirname(os.path.abspath(__file__)))

TOKEN_NUM = {
    'c4': 175,
    # 'github': 59,
    # 'book': 26,
    # 'arxiv': 28,
    # 'wikipedia': 24,
    # 'stackexchange': 20,
}


def get_split_ratio(target_token_num: int):
    total_token_num = sum(list(TOKEN_NUM.values()))
    ratio = float(target_token_num) / total_token_num
    print(f"Redpajama kept token ratio: {ratio:.2f}, target token num: {target_token_num}B")
    return ratio


def generate_new_url(url_path: str, save_path: str, ratio: float, category: str):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    random.seed(42)
    origin_txt = os.path.join(url_path, f"{category}.txt")
    new_txt = os.path.join(save_path, f"{category}.txt")
    with open(origin_txt, 'r') as f:
        origin_urls = f.readlines()
    origin_urls = random.sample(origin_urls, math.ceil(len(origin_urls) * ratio))
    for i in range(len(origin_urls)):
        if "\n" != origin_urls[i][-1]:
            origin_urls[i] = origin_urls[i] + '\n'
    with open(new_txt, 'w') as f:
        f.writelines(origin_urls)
    

if "__main__" == __name__:
    ratio = get_split_ratio(60)
    for c in TOKEN_NUM.keys():
        generate_new_url("./RedPajama-Data-1T/urls", "./new_urls", ratio, c)
    print("New urls generated and saved in ./new_urls")
