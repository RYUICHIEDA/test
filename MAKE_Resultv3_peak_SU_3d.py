import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import johnsonsu

# 光の波長をリストの番号に変える関数(中の数値はSpecimIQの取得する光波長の範囲と分解能に依存)
def Light_Lambda_to_Light_Lambda_number(Light_Lambda_value):
    return int(round((Light_Lambda_value - 397.32) / ((1003.58 - 397.32) / 203), 0))

# フォルダパスから指定した範囲のnpyファイルを読み込み、4次元データを作成
# 以下、返り値dataの要素
# 1: 写真番号インデックス
# 2: 光波長インデックス
# 3: 画像のxピクセル
# 4: 画像のyピクセル
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

# johnsonsu分布のPDF（確率密度関数）
def johnsonsu_pdf(x, loc, scale, a, b):
    return johnsonsu.pdf(x, a, b, loc=loc, scale=scale)

# 引数として入力したデータarrにSU分布のピークフィッティングを行い、算出したデータをグラフとして保存し、評価値を返す
def SU_fitting(arr, stlist, wavelength, save_path, photo_num):
    
    p0 = [np.argmax(arr),1,0,np.std(arr)]
    try:
        popt_johnsonsu, _ = curve_fit(johnsonsu_pdf, stlist, arr, p0=p0, maxfev=10000000)  # maxfevを増加させる
    except RuntimeError as e:
        print(f"Error fitting JohnsonSU distribution: {e}")
        return None
    
    fitted_data = johnsonsu_pdf(stlist, *popt_johnsonsu)

    # 誤差率の計算
    residuals = arr - fitted_data
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((arr-np.mean(arr))**2)
    r_squared = 1 - (ss_res/ss_tot)

    a_fit, b_fit, loc_fit, scale_fit = popt_johnsonsu
    
    # FWHMの計算
    half_value = np.max(fitted_data) / 2
    indices_above_half_max = np.where(fitted_data >= half_value)[0]
    FWHM_johnsonsu = stlist[indices_above_half_max[-1]] - stlist[indices_above_half_max[0]]
    
    # 13.5%幅の計算
    thre_value = np.max(fitted_data) / (np.e * np.e)
    indices_above_thre_max = np.where(fitted_data >= thre_value)[0]
    x_e2 = stlist[indices_above_thre_max[-1]] - stlist[indices_above_thre_max[0]]

    # 13.5%幅のピークから左側と右側の幅の比を計算
    x_center = stlist[np.argmax(fitted_data)]
    x_LS_Center = abs(x_center - stlist[indices_above_thre_max[0]])
    x_RS_Center = abs(x_center - stlist[indices_above_thre_max[-1]])
    x_balance = x_RS_Center / x_LS_Center * 100
    
    # フィッティングパラメータとFWHMの結果を返す
    return b_fit

xpixel = 512
ypixel = 512

y_center = 242

# ファイルのパス
folder_path = r"C:\Users\edaryuuichi\Desktop\240612\output"  # 000_processed.npyがあるフォルダのパスを指定してください

sp_st_num = 681    # 開始番号
sp_en_num = 697     # 終了番号

# 光の波長λ[nm]を指定するリスト
Light_Lambda = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

# 作成したグラフを保存するファイルパス
glaph_save_path = os.path.join(os.path.dirname(folder_path), f'{sp_st_num}_{sp_en_num}')
if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)
#Excel_path = os.path.join(glaph_save_path, f'SU_{sp_st_num}_{sp_en_num}.xlsx')

# スポット光照射のデータをsp_dataに格納
sp_data = process_files(sp_st_num, sp_en_num)

# 結果を保存するためのリスト
results = np.zeros((sp_data.shape[0],sp_data.shape[1],ypixel))

# 各波長に対してデータ処理を行う
for h in range(sp_data.shape[0]):
    for i in range(sp_data.shape[1]):
        for j in range(ypixel):
            wavelength = Light_Lambda[i]
            arr = sp_data[h, i, :, j]
            stlist = np.arange(0, 512)
            
            # フィッティングと結果の取得
            results[h,i,j] = SU_fitting(arr/np.sum(arr), stlist, wavelength, glaph_save_path, sp_st_num + h)

# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(0,17)

for h in range(sp_data.shape[1]):
    for i in range(ypixel):
        ax.plot(x,results[:,h,i],zs=i)
    ax.set_xlabel('No.')
    ax.set_ylabel('ERROR rate')
    ax.set_zlabel('ypixel')
    # グラフの表示
    plt.show()