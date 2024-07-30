'''
Auther : 江田龍宇一　Updated : 24/1/25 15:47

指定したフォルダ内にあるSpecimIQの生データとDARKREFを取得して、別途指定したフォルダに保存します。
その後、生データからDARKREFを引いたデータをnumpy形式で保存します。

主な操作はsource_pathと、input_folderの中身を変えるだけです。

ただし、フォルダのパスの中に日本語が入っているとうまく動作しないかもしれません。
そういった場合は、抽出したいデータが入っているフォルダをデスクトップなどコピーするとうまくいきます。
'''

import os
import shutil
import glob
import re
import spectral
import numpy as np

# 指定したフォルダからSpecimIQの撮影データを抽出して保存する
def copy_files_to_input_folder(source_folder, input_folder):
    def copy_file(file_path):
        destination_path = os.path.join(input_folder, os.path.basename(file_path))
        shutil.copy(file_path, destination_path)
        print(f'コピー: {os.path.basename(file_path)}')

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    for root, _, _ in os.walk(source_folder):
        for ext in ['hdr', 'raw']:
            search_pattern = os.path.join(root, f'[0-9][0-9][0-9].{ext}')
            for file_path in glob.glob(search_pattern):
                copy_file(file_path)

                darkref_file_path = os.path.join(root, f'DARKREF_{os.path.splitext(os.path.basename(file_path))[0]}.{ext}')
                if os.path.exists(darkref_file_path):
                    copy_file(darkref_file_path)

# 生データからDARKREFを引く処理
def custom_processing(data_x, data_darkref_x):
    # data_x の第二次元を変えながら処理を行う
    processed_data = data_x - data_darkref_x[0]
    return processed_data

# inputフォルダからファイルを取り出し、計算処理後outputフォルダにnumpy形式で保存する
def process_and_save_files(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # .hdr ファイルのリストを取得
    pattern = re.compile(r'^\d{3}\.hdr$')
    hdr_files = [f for f in os.listdir(input_folder) if pattern.match(f)]

    for hdr_file in hdr_files:
        # ファイルのパス
        input_file_path_x = os.path.join(input_folder, hdr_file)
        input_file_path_darkref_x = os.path.join(input_folder, f'DARKREF_{os.path.splitext(hdr_file)[0]}.hdr')

        # 計算処理の適用
        data_x = np.array(spectral.open_image(input_file_path_x).load())
        data_darkref_x = np.array(spectral.open_image(input_file_path_darkref_x).load())
        processed_data = custom_processing(data_x, data_darkref_x)

        # ファイル名に "_processed" を追加して保存
        output_file_name = f"{os.path.splitext(hdr_file)[0]}_processed.npy"
        output_file_path = os.path.join(output_folder, output_file_name)
        np.save(output_file_path, processed_data)

        print(f"処理が完了しました。保存先: {output_file_path}")

# フォルダのパス
source_path = r"C:\Users\edaryuuichi\Box\0_EDA_BOX\specimIQ data\240612"       # データを抽出するフォルダのパス
input_folder = r'C:\Users\edaryuuichi\Desktop\240612\input'     # 抽出したデータを保存する場所
output_folder = os.path.join(os.path.dirname(input_folder), 'output')

# フォルダの中身をコピーして処理
copy_files_to_input_folder(source_path, input_folder)
process_and_save_files(input_folder, output_folder)
