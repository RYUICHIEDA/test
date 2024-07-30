import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import johnsonsu

def emg(x, A, mu, sigma, lambda_):
    return A * np.exp(lambda_ * (lambda_ * sigma**2 / 2 - (x - mu))) * \
           0.5 * (1 + erf((x - mu - lambda_ * sigma**2) / (sigma * np.sqrt(2))))

# SU分布の定義
def su(x, A, mu, sigma, lambda_):
    return A * np.exp(-((x - mu) / sigma)**2 / 2) * \
           (1 + erf(lambda_ * (x - mu) / (sigma * np.sqrt(2))))

# EMGとSU分布の組み合わせ
def combined_model(x, A_emg, mu_emg, sigma_emg, lambda_emg, A_su, mu_su, sigma_su, lambda_su):
    return emg(x, A_emg, mu_emg, sigma_emg, lambda_emg) + su(x, A_su, mu_su, sigma_su, lambda_su)

def erf(x):
    # 恒等関数の近似を用いた誤差関数の実装
    # scipyのerf関数を使う方が簡単
    return np.sign(x) * np.sqrt(1 - np.exp(-x*x*(4/np.pi + 0.147*x*x)/(1 + 0.147*x*x)))

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

# 横軸を写真番号インデックスとして、指定したy座標のx軸最大値をプロットする
# ピーク値と波長の関係性を見るために作成
def plot_peak(arr, stlist, wavelength, save_path, photo_num):
    # johnsonsu分布フィッティング
    #p0 = [np.argmax(arr), np.var(arr), -1, np.max(arr)*10]  # 初期パラメータ [loc, scale, a, b]
    
    p0 = [np.argmax(arr),1,0,np.std(arr)]
    try:
        popt_johnsonsu, _ = curve_fit(johnsonsu_pdf, stlist, arr, p0=p0, maxfev=10000000)  # maxfevを増加させる
    except RuntimeError as e:
        print(f"Error fitting JohnsonSU distribution: {e}")
        return None
    
    fitted_data = johnsonsu_pdf(stlist, *popt_johnsonsu)

    '''
    initial_guess = [1,np.argmax(arr),np.std(arr),0,1,np.argmax(arr),1,0]
    popt, pcov = curve_fit(combined_model,stlist,arr,p0=initial_guess,maxfev=10000000)

    fitted_data = combined_model(stlist, *popt)
    '''

    # 元データのプロット
    plt.plot(stlist, arr, 'b-', label='Original Data')
    
    # johnsonsu分布のフィッティング後のデータのプロット
    plt.plot(stlist, fitted_data, 'g--', label='JohnsonSU Fit')
    
    plt.title(f'x方向スキャンのピーク値 \n 波長: {wavelength} nm', fontname='MS Gothic') 
    plt.xlabel('xピクセル [pixel]', fontname='MS Gothic')
    plt.ylabel('輝度', fontname='MS Gothic')
    #plt.xlim(0, 511)
    plt.legend()
    
    # グラフの保存
    save_filename = os.path.join(save_path, f'SU_{photo_num}_{wavelength}nm_johnsonsu_fit.png')
    plt.savefig(save_filename, bbox_inches='tight')  # bbox_inches='tight'で余白を最小限にして保存
    plt.close()  # グラフを閉じることでメモリを解放する

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

    x_center = stlist[np.argmax(fitted_data)]
    x_LS_Center = abs(x_center - stlist[indices_above_thre_max[0]])
    x_RS_Center = abs(x_center - stlist[indices_above_thre_max[-1]])
    x_balance = x_RS_Center / x_LS_Center * 100
    
    # フィッティングパラメータとFWHMの結果を返す
    return {
        'Param1': wavelength,
        'JohnsonSU_Max_Amplitude': np.max(fitted_data),
        'JohnsonSU_FWHM': FWHM_johnsonsu,
        'JohnsonSU_13.5%': x_e2,
        'JohnsonSU_BALANCE': x_balance,
        'SU_STD': a_fit,
        'SU_DIST': b_fit,
        'SU_MAX': scale_fit,
        'SU_POS': loc_fit,
        'ERR': r_squared
    }

xpixel = 512
ypixel = 512

y_center = 242

# ファイルのパス
folder_path = r"C:\Users\edaryuuichi\Desktop\240612\output"  # 000_processed.npyがあるフォルダのパスを指定してください

photo_num = 588

# 光の波長λ[nm]を指定するリスト
Light_Lambda = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

# 作成したグラフを保存するファイルパス
glaph_save_path = os.path.join(os.path.dirname(folder_path), f'{photo_num}')
if not os.path.exists(glaph_save_path): os.makedirs(glaph_save_path)
Excel_path = os.path.join(glaph_save_path, f'SU_{photo_num}.xlsx')

# スポット光照射のデータをsp_dataに格納
sp_data = process_files(photo_num, photo_num)

# 結果を保存するためのリスト
results = []

# 各波長に対してデータ処理を行う
for h in range(sp_data.shape[0]):
    for i in range(sp_data.shape[1]):
        wavelength = Light_Lambda[i]
        arr = sp_data[h, i, :, y_center]
        stlist = np.arange(0, 100)
        
        # フィッティングと結果の取得
        fit_results = plot_peak((arr[205:305]-np.min(arr[205:305]))/np.sum(arr[205:305]), stlist, wavelength, glaph_save_path, photo_num)
        if fit_results:
            # 結果をリストに追加
            results.append({
                'Param1': fit_results['Param1'],
                'Param2': f'No.{h}',
                'SU_Max_Amplitude': (fit_results['JohnsonSU_Max_Amplitude']+np.min(arr[205:305]))*np.sum(arr[205:305]),
                'SU_FWHM': fit_results['JohnsonSU_FWHM'],
                'SU_13.5%': fit_results['JohnsonSU_13.5%'],
                'SU_BALANCE': fit_results['JohnsonSU_BALANCE'],
                'SU_STD': fit_results['SU_STD'],
                'SU_DIST': fit_results['SU_DIST'],
                'SU_MAX': fit_results['SU_MAX'],
                'SU_POS': fit_results['SU_POS'],
                'ERROR_RATE': fit_results['ERR']
            })

# データフレームに変換
df_results = pd.DataFrame(results)

# データフレームをExcelファイルに書き込む
df_results.to_excel(Excel_path, index=False)

print("結果をエクセルファイルに保存しました。")
