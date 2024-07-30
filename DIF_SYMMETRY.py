import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
def process_files(folder_path, start_number, end_number, lamda, xpixel):
    g_data = np.zeros((end_number - start_number + 1, len(lamda), xpixel))

    for number in range(start_number, end_number + 1):
        # ファイル名を組み立て
        file_name = f"{number:03d}_processed.npy"
        file_path = os.path.join(folder_path, file_name)
        # ファイルが存在するか確認
        if os.path.exists(file_path):
            # ファイルの読み込み
            data = np.load(file_path)
            for lam in range(len(lamda)):
                g_data[number - start_number, lam, :] = data[:, y, lamda_to_lamdanumber(lamda[lam])]
    
    # rawデータだとx軸方向の情報が逆向きに保存されているため以下のコードで元に戻す
    rg_data = np.flip(g_data, axis=2)
    return rg_data

# 指定したピクセル数の範囲内で左右に分割した時の差分が最小となる
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

# グラフ描画（将来的にはy軸方向込みの分布図にしたい）
def draw_graph(arr):
    # 配列の要素の大きさに応じて色を変換する関数を定義
    norm = Normalize(vmin=arr.min(), vmax=arr.max())
    cmap = plt.get_cmap('Reds')
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # 四角の領域を作成し、色を変える
    for i in range(len(arr)):
        color = mapper.to_rgba(arr[i])
        # 凡例の名付け
        label_text = f'{steplist[i] * (-1) if steplist[i] != 0 else 0}mm'
        plt.fill([i, i+1, i+1, i], [0, 0, 1, 1], label=label_text, color=color)
    # タイトルと軸ラベル
    plt.title(f'血栓度分布？', fontname ='MS Gothic') 
    plt.xlabel('x [mm]')
    plt.ylabel('血栓度', fontname ='MS Gothic')
    # グラフの横軸の数値を変更する
    plt.xticks(np.arange(len(arr)) + 0.5, gca(len(arr),0.5))
    # 凡例
    legend = plt.legend(loc='center left', bbox_to_anchor=(1., .5))
    legend.set_title(f'スポット光位置',prop={"family":"MS Gothic"})
    # グラフの保存
    save_filename = f'{glaph_save_path}\\分布グラフ.png'
    plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存
    # グラフを表示
    plt.show()

# 以下に囲っている部分がユーザーが変更できる箇所
#----------------------------------------------------------------------------------------------------------------------------------

# ファイルのパス
output_folder = r'C:\Users\edaryuuichi\Box\0_EDA_BOX\program_PC_PC\data\2024-0201\output' # 000_processed.npyがあるフォルダのパスを指定してください
glaph_save_path = os.path.join(os.path.dirname(output_folder), 'glaph')

# ファイルの開始番号と終了番号（要書き換え）
start_number = 153   # ファイルの開始番号（測定では光源が測定対象中心の右側にあります）
end_number = 169    # ファイルの終了番号（測定では光源が測定対象中心の左側にあります）

step = 0.5          # ステップ距離[mm]
y = 225     # y座標　現状1点に固定しています

# 光の波長λ[nm]を指定するリスト　順不同でも可
lamda = [760]

center_range = 1
cal_range = 5

#----------------------------------------------------------------------------------------------------------------------------------

if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)

# 定数（書き換えないでください）
mpp = 0.157 # 1ピクセルあたりの長さ[mm]　画像から判読して決定しているため、後に再調査して変更する必要あり
xpixel = 512
dif_data = np.zeros((end_number - start_number + 1, len(lamda), xpixel//2))
integral_data = np.zeros((end_number - start_number + 1, len(lamda)))
x_axis = []
differ_min = []

steplist = gca(end_number - start_number + 1, step)

# 以下メイン処理

# 3次元データ作成
data = process_files(output_folder,start_number,end_number, lamda, xpixel)

for i in range(data.shape[0]):
    differ_min.append(dif_min(data[i,0,:], cent_ran=center_range, cal_ran=cal_range))

print(differ_min)

draw_graph(np.array(differ_min))