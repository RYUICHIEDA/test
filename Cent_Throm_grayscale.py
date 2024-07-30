import matplotlib.pyplot as plt
import os
import numpy as np

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

# numpy形式の4次元データを作成
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

# 輝度のカラーマップを描画するプログラム
def draw_grayscale(arr, lamda,xst,xen,yst,yen,start_number):
    # 配列の要素の大きさに応じて色を変換する関数を定義
    cmap = plt.get_cmap('gray')
    for i in range(arr.shape[0]):
        for j in range(len(lamda)):
            # グラフをプロット
            plt.imshow(arr[i,j,:,:].T, cmap=cmap ,extent=[xst, xen, yen, yst])

            # 軸ラベル（入れ替えたのでラベルも入れ替える）
            plt.title(f'グレースケール画像 \n (光波長[nm]　λ = {lamda[j]}) \n ファイル番号 : {start_number+i}', fontname ='MS Gothic') 
            plt.xlabel('x [pixel]')
            plt.ylabel('y [pixel]')

            plt.yticks(range(yst, yen+1, int((yen-yst)/10)))

            # グラフの保存
            save_filename = f'{glaph_save_path}\\No{start_number+i}_λ{lamda[j]}_グレースケール画像.png'
            plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存

            # グラフを表示
            plt.show()

# 以下に囲っている部分がユーザーが変更できる箇所
#----------------------------------------------------------------------------------------------------------------------------------

# ファイルのパス
output_folder = r'C:\Users\edaryuuichi\Desktop\240322\A\output' # 000_processed.npyがあるフォルダのパスを指定してください
# 作成したグラフを保存するファイルパス
glaph_save_path = os.path.join(os.path.dirname(output_folder), 'glaph')

# ファイルの開始番号と終了番号（要書き換え）
st_num =  290
en_num =  323

yst = 210           # yの領域の始点
yen = 250           # yの領域の終点

x_center = 265
LS = 3
RS = 5

# 光の波長λ[nm]を指定するリスト　順不同でも可
lamda = [550,640]

#----------------------------------------------------------------------------------------------------------------------------------

if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)

# 定数（書き換えないでください）
mpp = 0.15152 # 1ピクセルあたりの長さ[mm]　画像から判読して決定しているため、後に再調査して変更する必要あり

# 画像のxとyのピクセル数
xpixel = 512
ypixel = 512

xst = int(x_center - LS / mpp)
xen = int(x_center + RS / mpp)

# NumPy配列
data = process_files(output_folder,st_num,en_num,lamda,xpixel,ypixel)

# 画像を表示
draw_grayscale(data[:,:,xst:xen,yst:yen],lamda,xst,xen,yst,yen,st_num)