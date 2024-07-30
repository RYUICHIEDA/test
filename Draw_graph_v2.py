'''
Auther : 江田龍宇一　Updated : 24/1/25 17:00

指定したフォルダ内のデータを使ってグラフを描画します。
ステージをｘ軸線上でに走査した時の光強度の分布を可視化します。

高さｙを固定し、ｘ軸と光強度の二次元グラフを重ね描きします。
なお、光波長λは任意の値をリストに入れることでそれぞれ描き出します。

SpecimIQの仕様として撮影したデータの名前は3桁の番号を連続してつけていくため、
グラフを描く際には測定開始時の番号と測定終了時のファイル番号を指定する必要があります。
'''

import os
import numpy as np
import matplotlib.pyplot as plt

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
def process_files(folder_path, start_number, end_number, lamda):
    g_data = np.zeros((end_number - start_number + 1, len(lamda), 512))

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

# 血栓モデルを中心としたグラフを描画する際に、データのx軸の要素数を拡張する
# steplistとmppを用いて要素数を追加する　なお、steplistはファイルの開始・終了番号とステップ距離から生成する
# 光源が右側にあるデータは要素の後半にデータを格納しなおし、左側にあるデータは前半部にデータを格納する
# 空いている要素の部分は光強度が０としている
def extend_array(glaph_data, steplist, mpp):
    step = steplist[(len(steplist)//2) + 1]
    im_diff = int((len(steplist) - 1) * step / mpp)
    ex_len = glaph_data.shape[2] + im_diff
    extended_array = np.zeros((glaph_data.shape[0], glaph_data.shape[1], ex_len))

    for number in range(glaph_data.shape[0]):
        move = int(step / mpp * number)
        left_shift = im_diff - move
        right_shift = left_shift + glaph_data.shape[2]
        extended_array[number, :, left_shift:right_shift] = glaph_data[number, :, :]
    return extended_array

# グラフを描画する
# グラフは光波長λごとに描画して、glaph_save_pathというフォルダに保存する
def graph_draw(glaph_data, lamda, glaph_save_path, steplist, mpp, cap1, cap2, start, end):
    for lam in range(glaph_data.shape[1]):
        for number in range(glaph_data.shape[0]):
            # グラフ描画の色設定（暫定）
            color_value = number / (glaph_data.shape[0] - 1)
            line_color = plt.cm.viridis(color_value)
            # 凡例の名付け
            label_text = f'{steplist[number] * (-1) if steplist[number] != 0 else 0}mm'
            plt.plot(gca(glaph_data.shape[2], mpp), glaph_data[number, lam, :], marker=None, label=label_text, color=line_color)
  
        # タイトルと軸ラベル
        plt.title(f' {cap1}位置を中心にプロット　（{cap1}位置　x = 0） \n (光波長[nm]　λ = {lamda[lam]}) \n ファイル番号 : {start} - {end}', fontname ='MS Gothic') 
        plt.xlabel('x [mm]')
        plt.ylabel('光強度', fontname ='MS Gothic')
        # 各軸の描写範囲
        plt.xlim(-6, 6)
        plt.ylim(0, np.max(glaph_data[:, lam, :])*1.05)
        # 各軸の目盛の刻み幅指定 後々y軸も追加する
        plt.xticks(np.arange(-6, 7, 1))
        # 凡例
        legend = plt.legend(loc='center left', bbox_to_anchor=(1., .5))
        legend.set_title(f'{cap2}位置',prop={"family":"MS Gothic"})
        # グラフの保存
        save_filename = f'{glaph_save_path}\\x0_{cap1}中心_lambda_{lamda[lam]}.png'
        plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存

        plt.show()

# 以下に囲っている部分がユーザーが変更できる箇所
#----------------------------------------------------------------------------------------------------------------------------------

# ファイルのパス
output_folder = r"C:\Users\edaryuuichi\Desktop\240322\A\output" # 000_processed.npyがあるフォルダのパスを指定してください
glaph_save_path = os.path.join(os.path.dirname(output_folder), 'glaph')

# ファイルの開始番号と終了番号（要書き換え）
start_number = 315   # ファイルの開始番号（測定では光源が測定対象中心の右側にあります）
end_number = 323    # ファイルの終了番号（測定では光源が測定対象中心の左側にあります）

step = 1          # ステップ距離[mm]
y = 230     # y座標　現状1点に固定しています

# 光の波長λ[nm]を指定するリスト　順不同でも可
lamda = [525,660,810,940]

#----------------------------------------------------------------------------------------------------------------------------------

if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)

# 定数（書き換えないでください）
mpp = 0.157 # 1ピクセルあたりの長さ[mm]　画像から判読して決定しているため、後に再調査して変更する必要あり

steplist = gca(end_number - start_number + 1, step)

# メイン処理

# 3次元データ作成
data = process_files(output_folder,start_number,end_number, lamda)

# 光源中心のグラフ描画
#graph_draw(data, lamda, glaph_save_path, steplist, mpp, start=start_number, end=end_number, cap1='ライト照射', cap2='血栓')

# モデル中心のグラフ描画
graph_draw(extend_array(data,steplist,mpp), lamda, glaph_save_path, steplist, mpp, cap1='血栓', cap2='ライト照射', start=start_number, end=end_number)