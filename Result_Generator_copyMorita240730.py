import numpy as np
import io
import os
from scipy.stats import johnsonsu
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
from matplotlib.colors import LogNorm
from multiprocessing import Pool, cpu_count

# グローバル変数
global folder_path
global glaph_save_path
global Light_Lambda
global xpixel, ypixel
global mpp
global sp_xscan_num, sp_yscan_num
global y_center
global pix_sp_yrange
global yst, yen
global steplist
global row

# -LSからRSの範囲までdeltaごとに区切ったリストを作成
def make_step(LS, RS, delta):
    lst = []
    lstlen = int((LS + RS) / delta) + 1
    for i in range(lstlen):
        lst.append(-LS + i * delta)
    return lst

# 光の波長をリストの番号に変える関数(中の数値はSpecimIQの取得する光波長の範囲と分解能に依存)
def Light_Lambda_to_Light_Lambda_number(Light_Lambda_value):
    return int(round((Light_Lambda_value - 397.32) / ((1003.58 - 397.32) / 203), 0))

# 指定したピクセル数の範囲内で左右に分割した時の差分の総和を返す関数
def dif(data, x_center, cal_ran):
    dif = 0
    for i in range(int((cal_ran / 2) / mpp)):
        L_index = x_center - 1 - i
        R_index = x_center + i
        dif += abs(data[L_index] - data[R_index])
    return dif

# ファイルを処理する関数
def process_file(data_index):
    data = np.zeros((len(Light_Lambda), xpixel, ypixel))
    file_path = os.path.join(folder_path, f"{data_index:03d}_processed.npy")
    if os.path.exists(file_path):
        l_data = np.load(file_path)
        for lam, Light_Lambda_value in enumerate(Light_Lambda):
            data[lam, :, :] = l_data[:, :, Light_Lambda_to_Light_Lambda_number(Light_Lambda_value)]
    return data

# フォルダパスから指定した範囲のnpyファイルを読み込み、4次元データを作成
def process_files(st_num, en_num):
    data = np.zeros((en_num - st_num + 1, len(Light_Lambda), xpixel, ypixel))
    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, range(st_num, en_num + 1))
    for i, result in enumerate(results):
        data[i] = result
    data = np.flip(data, axis=2)
    return data

# データを処理する関数
def process_data_chunk(args):
    data_chunk, x_center, cal_ran, method = args
    eva_data = np.zeros((data_chunk.shape[0], data_chunk.shape[1], data_chunk.shape[3]))
    for i in range(data_chunk.shape[0]):
        for j in range(data_chunk.shape[1]):
            for k in range(data_chunk.shape[3]):
                if method == 0:
                    eva_data[i, j, k] = np.mean(data_chunk[i, j, x_center - int(cal_ran / mpp):x_center + int(cal_ran / mpp), k])
                elif method == 1:
                    eva_data[i, j, k] = data_chunk[i, j, x_center, k]
                elif method == 2:
                    eva_data[i, j, k] = dif(data_chunk[i, j, :, k], x_center, cal_ran)
    return eva_data

# 4次元データdataを指定したmethodで処理し、3次元の評価データを出力する関数
def process_data(data, x_center, cal_ran, method):
    num_chunks = cpu_count()
    chunk_size = data.shape[0] // num_chunks
    chunks = [(data[i * chunk_size:(i + 1) * chunk_size], x_center, cal_ran, method) for i in range(num_chunks)]
    with Pool(cpu_count()) as pool:
        results = pool.map(process_data_chunk, chunks)
    eva_data = np.concatenate(results, axis=0)
    return eva_data

# 以下、その他の関数（省略）

# 定数（書き換えないでください）
mpp = 0.15152 # 1ピクセルあたりの長さ[mm]　画像から判読して決定しているため、後に再調査して変更する必要あり

# 画像のxとyのピクセル数
xpixel = 512
ypixel = 512

#row = 2         # Excelの行の初期位置
column = 2      # Excelの列の位置（未実装）
wb = Workbook()
ws = wb.active

# 以下に囲っている部分がユーザーが変更できる箇所
#----------------------------------------------------------------------------------------------------------------------------------

# 全光照射画像のファイル番号
ensu_num =  588

# スポット光照射画像のファイル番号
sp_st_num = 589     # 開始番号
sp_en_num = 605     # 終了番号

# 撮影条件

sp_xscan_num = 17   # x方向の撮影枚数
sp_yscan_num = 1    # y方向の撮影枚数

deltax = 0.5        # xのステップ区切り[mm]
deltay = 1          # yのステップ区切り[mm]（未実装）

UPS = 0             # y_centerから上方向の距離[mm]
UNS = 0             # y_centerから下方向の距離[mm]

LS = 4              # x_centerから左方向の距離[mm]
RS = 4              # x_centerから右方向の距離[mm]

yst = 220           # yの領域の始点
yen = 260           # yの領域の終点

# 描画範囲など

x_center = 262      # xpixelの中心位置
y_center = 242      # ypixelの中心位置

sp_yrange = 1.5     # y軸方向にカラーマップを重ね合わせる時の取得する範囲[mm]

SU_yrange = 5

# 光の波長λ[nm]を指定するリスト　順不同でも可
Light_Lambda = [400,450,500,550,600,650,700,750,800,850,900,950,1000]

#---------------------------------------------------------------------------------------------------------------------------------
# ファイルのパス
folder_path = r"C:\Users\nobutomomorita\Box\Personal_N17474\0_EDA_BOX\program_PC_PC\data\output" # 000_processed.npyがあるフォルダのパスを指定してください

# 作成したグラフを保存するファイルパス
glaph_save_path = os.path.join(os.path.dirname(folder_path), f'SU_Results_s{sp_st_num}_e{sp_en_num}')
if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)
Excel_Param_Path = os.path.join(glaph_save_path, f'SU_Param_{sp_st_num}_{sp_en_num}.xlsx')
Excel_Map_Path = os.path.join(glaph_save_path, f'SU_MAP_{sp_st_num}_{sp_en_num}.xlsx')

def main():
    row = 2
    # スポット光照射のデータをsp_dataに格納
    sp_data = process_files(sp_st_num, sp_en_num)

    pix_sp_yrange = int(sp_yrange/mpp)  # sp_yrangeを...

    # 評価データの処理
    eva_data = process_data(sp_data, x_center, LS + RS, 0)

    # 以下、その他の処理（省略）

if __name__ == "__main__":
    main()
