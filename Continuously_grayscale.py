import matplotlib.pyplot as plt
import os
import numpy as np



global folder_path
global Light_Lambda
global xpixel
global ypixel



# 光の波長をリストの番号に変える関数(中の数値はSpecimIQの取得する光波長の範囲と分解能に依存)
def Light_Lambda_to_Light_Lambda_number(Light_Lambda_value):
    return int(round((Light_Lambda_value - 397.32) / ((1003.58 - 397.32) / 203), 0))

# フォルダパスから指定した範囲のnpyファイルを読み込み、4次元データを作成
# 第一要素は走査数（光源の位置が右から左の順に格納）、第二要素は任意の光波長λ
# 第三要素はx軸のピクセル、格納されるデータは光強度   
def process_files(st_num, en_num):
    data = np.zeros((en_num - st_num + 1, len(Light_Lambda), xpixel, ypixel))
    
    for number, data_index in enumerate(range(st_num, en_num + 1)):
        file_path = os.path.join(folder_path, f"{data_index:03d}_processed.npy")
        if os.path.exists(file_path):
            l_data = np.load(file_path)
            for lam, Light_Lambda_value in enumerate(Light_Lambda):
                data[number, lam, :, :] = l_data[:, :, Light_Lambda_to_Light_Lambda_number(Light_Lambda_value)]
    
    data = np.flip(data, axis=2)
    
    return data

# 輝度のカラーマップを描画するプログラム
def draw_grayscale(arr, file_num, xst, xen, yst, yen, cap):
    # 配列の要素の大きさに応じて色を変換する関数を定義
    cmap = plt.get_cmap('gray')
    for j in range(len(Light_Lambda)):
        arr_normalized = (arr[j,:,:] - np.min(arr[j,:,:])) / (np.max(arr[j,:,:]) - np.min(arr[j,:,:]))
        # グラフをプロット
        plt.imshow(arr_normalized.T, cmap=cmap ,extent=[xst, xen, yen, yst])

        # 軸ラベル（入れ替えたのでラベルも入れ替える）
        plt.title(f'{cap}グレースケール画像 \n (光波長[nm]　λ = {Light_Lambda[j]}) \n ファイル番号 : {file_num}', fontname ='MS Gothic') 
        plt.xlabel('x [pixel]')
        plt.ylabel('y [pixel]')

        plt.yticks(range(yst, yen+1, int((yen-yst)/10)))

        # カラーバー
        plt.colorbar(label='Light Intencity')

        # グラフの保存
        save_filename = f'{glaph_save_path}\\No{file_num}_λ{Light_Lambda[j]}_{cap}グレースケール画像.png'
        plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存

        # グラフを表示
        plt.show()



# 定数（書き換えないでください）
mpp = 0.15152 # 1ピクセルあたりの長さ[mm]　画像から判読して決定しているため、後に再調査して変更する必要あり

# 画像のxとyのピクセル数
xpixel = 512
ypixel = 512



# 以下に囲っている部分がユーザーが変更できる箇所
#----------------------------------------------------------------------------------------------------------------------------------

# ファイルのパス
folder_path = r'C:\Users\edaryuuichi\Desktop\240322\D\output' # 000_processed.npyがあるフォルダのパスを指定してください
# 作成したグラフを保存するファイルパス
glaph_save_path = os.path.join(os.path.dirname(folder_path), 'glaph')
if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)

# 全光照射画像のファイル番号
ensu_num =  357

# スポット光照射画像のファイル番号
sp_st_num = 358     # 開始番号
sp_en_num = 375     # 終了番号

yst = 210           # yの領域の始点
yen = 250           # yの領域の終点

sp_xscan_num = 9
sp_yscan_num = int((sp_en_num - sp_st_num + 1) / sp_xscan_num)

x_center = 265      # xpixelの中心位置
LS = 3              # x_centerから左方向の距離[mm]
RS = 5              # x_centerから右方向の距離[mm]

# 光の波長λ[nm]を指定するリスト　順不同でも可
Light_Lambda = [550,640]

#----------------------------------------------------------------------------------------------------------------------------------



# NumPy配列にデータを読み込む
ensu_data = process_files(ensu_num, ensu_num)

# LSとRSで指定した距離[mm]分x軸を動かしたときのピクセルを計算
ensu_xst = int(x_center - LS / mpp)
ensu_xen = int(x_center + RS / mpp)

# 画像を表示
#draw_grayscale(ensu_data[0, :, ensu_xst:ensu_xen, yst:yen], ensu_num, ensu_xst, ensu_xen, yst, yen, cap = '全光照射')

sp_data = process_files(sp_st_num, sp_en_num)

#for i in range(sp_yscan_num):
for j in range(sp_xscan_num):
    sp_xst = int(x_center - LS / mpp - (RS - j) / mpp)
    sp_xen = int(x_center + RS / mpp - (RS - j) / mpp)
    draw_grayscale(sp_data[j, :, sp_xst:sp_xen, yst:yen], sp_st_num + j, sp_xst, sp_xen, yst, yen, cap = 'スポット光照射')
