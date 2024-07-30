import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

def create_folder(folder_path,folder_name):
    full_path = os.path.join(os.path.dirname(folder_path),folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

# 光の波長をリストの番号に変える関数(中の数値はSpecimIQの取得する光波長の範囲と分解能に依存)
def lamda_to_lamdanumber(lamda):
    return int(round((lamda - 397.32) / ((1003.58 - 397.32) / 203), 0))

# 配列lstの長さで配列の中心の要素を0とし、increment分だけ左右に増減した値を生成する
# 図のx軸の中心を０mmにしたいため作成
def gca(lst, increment):
    centered_array = []
    middle_index = lst // 2

    if lst % 2 == 1:  # リストの長さが奇数の場合
        for i in range(lst):
            centered_array.append(round((i - middle_index) * increment, 3))
    else:  # リストの長さが偶数の場合
        for i in range(lst):
            centered_array.append(round((i - middle_index + 0.5) * increment, 3))

    return centered_array

# numpy形式の3次元データを作成
# 第一要素は走査数（光源の位置が右から左の順に格納）、第二要素は任意の光波長λ
# 第三要素はx軸のピクセル、格納されるデータは光強度   
def process_files(folder_path, start_number, end_number, lamda, xpixel, ypixel):
    g_data = np.zeros((end_number - start_number + 1, len(lamda), xpixel, ypixel))
    
    for number, data_index in enumerate(range(start_number, end_number + 1)):
        file_path = os.path.join(folder_path, f"{data_index:03d}_processed.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)
            for lam, lamda_value in enumerate(lamda):
                g_data[number, lam, :, :] = data[:, :, lamda_to_lamdanumber(lamda_value)]
    
    return np.flip(g_data, axis=2)

'''
# 指定したピクセル数の範囲内で左右に分割した時の差分が最小となったときの値を返す関数
def dif_min(data,cent_ran,cal_ran):
    dif_list = []
    cent_ran_mp = int(2*cent_ran/mpp)
    cal_ran_mp = int(cal_ran/mpp)
    for i in range(cent_ran_mp):
        dif = 0
        for j in range(cal_ran_mp):
            L_index = len(data)//2 - 1 - cent_ran_mp//2 + i - j
            R_index = len(data)//2 - cent_ran_mp//2 + i + j
            dif += abs(data[L_index] - data[R_index])
        #print(dif) 
        dif_list.append(dif)
    return min(dif_list)
'''

# 指定したピクセル数の範囲内で左右に分割した時の差分の総和を返す関数
def dif_min2(data,cal_ran):
    dif = 0
    cal_ran_mp = int(cal_ran/mpp)
    for i in range(cal_ran_mp):
        L_index = len(data)//2 - 1 - i
        R_index = len(data)//2 + i
        dif += abs(data[L_index] - data[R_index])
    return dif

# 評価値を計算し、配列に格納する関数
# xy軸に応じて条件分岐して処理します(x:axis=0, y:axis=1)
def process_dif(data, axis):
    differ_min = np.zeros((data.shape[0],data.shape[1],data.shape[3 if axis==0 else 2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if axis==0:
                for k in range(data.shape[3]):
                    differ_min[i,j,k]=(dif_min2(data[i,j,:,k],cal_ran=cal_range))
            else:
                for k in range(data.shape[2]):
                    differ_min[i,j,k]=(dif_min2(data[i,j,k,:],cal_ran=data.shape[3]/4*mpp))
    return differ_min

# 評価値を計算し、配列に格納する関数
# xy軸に応じて条件分岐して処理します(x:axis=0, y:axis=1)
def aaa_dif(data, axis):
    differ_min = np.zeros((data.shape[0],data.shape[1],data.shape[3 if axis==0 else 2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if axis==0:
                for k in range(data.shape[2]):
                    for l in range(data.shape[3]):
                        differ_min[i,j,k]=data[i,j,k,l]
    return differ_min

def draw_graph(arr, stlist, st, en, st_num, en_num, lamda):
    # 配列の要素の大きさに応じて色を変換する関数を定義
    cmap = plt.get_cmap('Reds')

    for i in range(arr.shape[1]):
        # グラフをプロット
        plt.imshow(np.flip(arr[:,i,:], axis=0).T, cmap=cmap, origin='lower', interpolation='nearest', aspect='auto', extent=[stlist[0], stlist[-1], en, st],norm=LogNorm())

        # 軸ラベル（入れ替えたのでラベルも入れ替える）
        plt.title(f' 血栓位置を中心にプロット　（血栓位置　x = 0） \n (光波長[nm]　λ = {lamda[i]}) \n ファイル番号 : {st_num} - {en_num}', fontname ='MS Gothic') 
        plt.xlabel('スポット光位置 [mm]', fontname ='MS Gothic')
        plt.ylabel('y [pixel]')

        # カラーバー
        plt.colorbar(label='BC level')

        # グラフの保存
        save_filename = f'{glaph_save_path}\\No{st_num}-{en_num}_λ{lamda[i]}_x方向二次元カラーマップ.png'
        plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存

        # グラフを表示
        plt.show()

def ydraw_graph(arr, stlist, st, en, st_num, en_num, lamda):
    # 配列の要素の大きさに応じて色を変換する関数を定義
    cmap = plt.get_cmap('Reds')

    for i in range(arr.shape[1]):
        # グラフをプロット
        plt.imshow(np.flip(arr[:,i,:],axis=0), cmap=cmap, origin='lower', interpolation='nearest', aspect='auto',extent=[en, st, stlist[0], stlist[-1]],norm=plt.LogNorm())

        # 軸ラベル（入れ替えたのでラベルも入れ替える）
        plt.title(f' 血栓位置を中心にプロット　（血栓位置　y = 0） \n (光波長[nm]　λ = {lamda[i]}) \n ファイル番号 : {st_num} - {en_num}', fontname ='MS Gothic') 
        plt.xlabel('x[pixel]')
        plt.ylabel('スポット光位置 [mm]',fontname ='MS Gothic')

        # カラーバー
        plt.colorbar(label='BC level', format=LogFormatter())

        # グラフの保存
        save_filename = f'{glaph_save_path}\\No{st_num}-{en_num}_λ{lamda[i]}_y方向二次元カラーマップ.png'
        plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存

        # グラフを表示
        plt.show()




# 以下に囲っている部分がユーザーが変更できる箇所
#----------------------------------------------------------------------------------------------------------------------------------

# ファイルのパス
output_folder = r'C:\Users\edaryuuichi\Desktop\240322\A\output' # 000_processed.npyがあるフォルダのパスを指定してください
glaph_save_path = os.path.join(os.path.dirname(output_folder), 'glaph')

# ファイルの開始番号と終了番号（要書き換え）
start_number =  297  # ファイルの開始番号（測定では光源が測定対象中心の右側にあります）
end_number = 305    # ファイルの終了番号（測定では光源が測定対象中心の左側にあります）

step = 1          # ステップ距離[mm]

yst = 200           # yの領域の始点
yen = 250           # yの領域の終点

xst = 235           # xの領域の始点
xen = 280           # xの領域の終点

# 光の波長λ[nm]を指定するリスト　順不同でも可
lamda = [550,640,700,750,800]

center_range = 1
cal_range = 2

#----------------------------------------------------------------------------------------------------------------------------------

if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)

# 定数（書き換えないでください）
mpp = 0.157 # 1ピクセルあたりの長さ[mm]　画像から判読して決定しているため、後に再調査して変更する必要あり
xpixel = 512
ypixel = 512

#steplist = gca(end_number - start_number + 1, step)
steplist = range(-3,6)

# 以下メイン処理

# 4次元データ作成
data = process_files(output_folder,start_number,end_number, lamda, xpixel, ypixel)

#yscan = process_dif(data[:,:,:,yst:yen], axis=0)
# xscan = process_dif(data[:,:,xst:xen,:], axis=1)
yscan = aaa_dif(data[:,:,:,yst:yen], axis=0)

allnorms = np.zeros((yscan.shape))

# 全体正規化
for i in range(yscan.shape[1]):
    allnorms[:,i,:] = (yscan[:,i,:] - np.min(yscan[:,i,:])) / (np.max(yscan[:,i,:]) - np.min(yscan[:,i,:]))

xallnorms = (xscan - np.min(xscan)) / (np.max(xscan) - np.min(xscan))

draw_graph(allnorms, steplist, yst, yen, start_number, end_number, lamda)
#ydraw_graph(xallnorms, steplist, xst, xen, start_number, end_number, lamda)
