import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# フォルダが存在しないときに、フォルダを作成する関数
def create_folder(folder_path,folder_name):
    #フォルダ名の定義
    full_path = os.path.join(os.path.dirname(folder_path),folder_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path

# 光の波長をリストの番号に変える関数(中の数値はSpecimIQの取得する光波長の範囲と分解能に依存)
def lamda_to_lamdanumber(lamda):
    return int(round((lamda - 397.32) / ((1003.58 - 397.32) / 203), 0))

# numpy形式の3次元データを作成
# 第一要素は走査数（光源の位置が右から左の順に格納）、第二要素は任意の光波長λ
# 第三要素はx軸のピクセル、格納されるデータは光強度   
def process_files(folder_path, start_number, end_number, lamda, xpixel, ypixel):
    data = np.zeros((end_number - start_number + 1, len(lamda), xpixel, ypixel))
    
    for number, data_index in enumerate(range(start_number, end_number + 1)):
        file_path = os.path.join(folder_path, f"{data_index:03d}_processed.npy")
        if os.path.exists(file_path):
            l_data = np.load(file_path)
            for lam, lamda_value in enumerate(lamda):
                data[number, lam, :, :] = l_data[:, :, lamda_to_lamdanumber(lamda_value)]
    
    #data = np.flip(g_data, axis=3)
    data = np.flip(data, axis=2)
    
    return data

# 指定したピクセル数の範囲内で左右に分割した時の差分の総和を返す関数
# 24/04/16現在、スポット光中心がx=265[pixel]だったため、処理を変更しています コメントアウトしている部分が本来の処理
def dif_min(data,cal_ran):
    dif = 0
    cal_ran_mp = int((cal_ran/2)/mpp)
    for i in range(cal_ran_mp):
        L_index = 265 - 1 - i
        R_index = 265 + i
        #L_index = len(data)//2 - 1 - i
        #R_index = len(data)//2 + i
        dif += abs(data[L_index] - data[R_index])
    return dif

# 評価値を計算し、配列に格納する関数
# xy軸に応じて条件分岐して処理します(x:axis=0, y:axis=1)
def process_dif(data):
    # ３次元で要素がすべて０の格納用配列を作成
    # 第一要素:写真番号インデックス
    # 第二要素:光波長インデックス
    # 第三要素:yピクセル番号
    differ_min = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[4]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                for l in range(data.shape[4]):
                    differ_min[i,j,k,l]=(dif_min(data[i,j,k,:,l],cal_ran=cal_range))
    return differ_min

# 差分のカラーマップを描画するプログラム
def draw_dif_colormap(arr, stlist, st, en, st_num, en_num, lamda,x_scan):
    # 配列の要素の大きさに応じて色を変換する関数を定義
    cmap = plt.get_cmap('Reds_r')
    
    for i in range(arr.shape[0]):
        for j in range(arr.shape[2]):
            # グラフをプロット
            plt.imshow(arr[i,:,j,:].T, cmap=cmap, origin='lower', interpolation='nearest', aspect='auto', extent=[stlist[0], stlist[-1], en, st],norm=LogNorm())

            # 軸ラベル（入れ替えたのでラベルも入れ替える）
            plt.title(f'差分カラーマップ (y = {0-i}) \n 血栓位置を中心にプロット　（血栓位置　x = 0） \n (光波長[nm]　λ = {lamda[j]}) \n ファイル番号 : {st_num+i*x_scan} - {st_num+x_scan-1+i*x_scan}', fontname ='MS Gothic') 
            plt.xlabel('スポット光位置 [mm]', fontname ='MS Gothic')
            plt.ylabel('y [pixel]')

            plt.yticks(range(st, en+1, int((en-st)/10)))

            # カラーバー
            plt.colorbar(label='BC level')

            # グラフの保存
            save_filename = f'{glaph_save_path}\\Y{0-i}_No{st_num+i*x_scan}-{st_num+x_scan-1+i*x_scan}_λ{lamda[j]}_差分カラーマップ.png'
            plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存

            # グラフを表示
            plt.show()

# 輝度のカラーマップを描画するプログラム
def draw_max_colormap(arr, stlist, st, en, st_num, en_num, lamda,x_scan):
    # 配列の要素の大きさに応じて色を変換する関数を定義
    cmap = plt.get_cmap('Reds_r')

    for i in range(arr.shape[0]):
        for j in range(arr.shape[2]):
            # グラフをプロット
            plt.imshow(arr[i,:,j,:].T, cmap=cmap, origin='lower', interpolation='nearest',aspect='auto', extent=[stlist[0], stlist[-1], en, st],norm=LogNorm())

            # 軸ラベル（入れ替えたのでラベルも入れ替える）
            plt.title(f'輝度カラーマップ (y = {0-i})\n 血栓位置を中心にプロット　（血栓位置　x = 0） \n (光波長[nm]　λ = {lamda[j]}) \n ファイル番号 : {st_num+i*x_scan} - {st_num+x_scan-1+i*x_scan}', fontname ='MS Gothic') 
            plt.xlabel('スポット光位置 [mm]', fontname ='MS Gothic')
            plt.ylabel('y [pixel]')

            plt.yticks(range(st, en+1, int((en-st)/10)))

            # カラーバー
            plt.colorbar(label='Light Intencity')

            # グラフの保存
            save_filename = f'{glaph_save_path}\\Y{0-i}_No{st_num+i*x_scan}-{st_num+x_scan-1+i*x_scan}_λ{lamda[j]}_輝度カラーマップ.png'
            plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存

            # グラフを表示
            plt.show()


# 以下に囲っている部分がユーザーが変更できる箇所
#----------------------------------------------------------------------------------------------------------------------------------

# ファイルのパス
output_folder = r'C:\Users\edaryuuichi\Desktop\240322\D\output' # 000_processed.npyがあるフォルダのパスを指定してください
# 作成したグラフを保存するファイルパス
glaph_save_path = os.path.join(os.path.dirname(output_folder), 'glaph')

# ファイルの開始番号と終了番号（要書き換え）
start_number =  358 # ファイルの開始番号（測定では光源が測定対象中心の右側にあります）
end_number = 375    # ファイルの終了番号（測定では光源が測定対象中心の左側にあります）

xscan_num = 9
yscan_num = int((end_number - start_number + 1) / xscan_num)

# step = 1            # ステップ距離[mm]

yst = 210           # yの領域の始点
yen = 250           # yの領域の終点

# 光の波長λ[nm]を指定するリスト　順不同でも可
lamda = [550,640,700,750,800]

cal_range = 2       # xの中心位置から差分をとる範囲[mm]

#----------------------------------------------------------------------------------------------------------------------------------

if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)

# 定数（書き換えないでください）
mpp = 0.15152 # 1ピクセルあたりの長さ[mm]　画像から判読して決定しているため、後に再調査して変更する必要あり

# 画像のxとyのピクセル数
xpixel = 512
ypixel = 512

# 写真番号の位置を示す配列　カラーマップ描画時に使用
steplist = range(-3,6)

# 以下メイン処理

# 4次元データ作成
data = process_files(output_folder,start_number,end_number, lamda, xpixel, ypixel)
# data[i, j, k, l]の定義
#  要素i：写真番号インデックス
#  要素j：波長インデックス
#  要素k：ｘピクセル番号
#  要素l：yピクセル番号

scan_data = np.zeros((yscan_num,xscan_num,data.shape[1],data.shape[2],data.shape[3]))

for i in range(yscan_num):
    for j in range(xscan_num):
        scan_data[i, j, :, :, :] = data[i * xscan_num + j, :, :, :]

# 走査画像番号を逆向きにする（スポット光を測定対象の右から左にかけて動かした場合）
scan_data = np.flip(scan_data,axis=1)

# 要素i,j,lごとに差分の総和を格納
yscan = process_dif(scan_data[:,:,:,:,yst:yen])

# 要素j（光波長）ごとにyscanをmin-maxで正規化
for i in range(yscan.shape[0]):
    for j in range(yscan.shape[1]):
        for k in range(yscan.shape[2]):
            yscan[i,j,k,:] = (yscan[i,j,k,:] - np.min(yscan[i,j,k,:])) / (np.max(yscan[i,j,k,:]) - np.min(yscan[i,j,k,:]))

# 差分カラーマップ描画
draw_dif_colormap(yscan, steplist, yst, yen, start_number, end_number, lamda,xscan_num)

# 要素j（光波長）ごとにyscanをmin-maxで正規化
for i in range(scan_data.shape[0]):
    for j in range(scan_data.shape[1]):
        for k in range(scan_data.shape[2]):
            scan_data[i,j,k,265,yst:yen] = (scan_data[i,j,k,265,yst:yen] - np.min(scan_data[i,j,k,265,yst:yen])) / (np.max(scan_data[i,j,k,265,yst:yen]) - np.min(scan_data[i,j,k,265,yst:yen]))

# 輝度カラーマップ描画
draw_max_colormap(scan_data[:,:,:,265,yst:yen], steplist, yst, yen, start_number, end_number, lamda,xscan_num)