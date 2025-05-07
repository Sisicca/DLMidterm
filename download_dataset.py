import os
import requests
import tarfile
from tqdm import tqdm
import shutil

def download_file(url, filename):
    """
    下载文件
    
    参数:
        url: 文件URL
        filename: 保存文件名
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f'下载 {url.split("/")[-1]}'
    )
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

def extract_file(filename, extract_dir):
    """
    解压文件
    
    参数:
        filename: 压缩文件名
        extract_dir: 解压目录
    """
    print(f"解压 {filename} 到 {extract_dir}")
    with tarfile.open(filename) as tar:
        tar.extractall(path=extract_dir)

def download_caltech101():
    """
    下载Caltech-101数据集
    """
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 下载数据集
    url = 'https://data.caltech.edu/tindfiles/serve/e41f5188-0b32-41fa-801b-d1e840915e80/'
    filename = 'data/caltech101.tar.gz'
    
    if not os.path.exists(filename):
        print("下载Caltech-101数据集...")
        download_file(url, filename)
    else:
        print("Caltech-101数据集已存在，跳过下载")
    
    # 解压数据集
    extract_dir = 'data/temp'
    os.makedirs(extract_dir, exist_ok=True)
    
    if not os.path.exists('data/caltech101'):
        extract_file(filename, extract_dir)
        
        # 移动文件到正确位置
        shutil.move(os.path.join(extract_dir, '101_ObjectCategories'), 'data/caltech101')
        
        # 清理临时目录
        shutil.rmtree(extract_dir)
    else:
        print("Caltech-101数据集已解压，跳过解压")
    
    print("Caltech-101数据集准备完成")

if __name__ == '__main__':
    download_caltech101() 