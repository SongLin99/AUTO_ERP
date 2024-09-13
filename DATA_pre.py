import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from autoreject import Ransac  # noqa
from autoreject.utils import interpolate_bads  # noqa
import autoreject
def select_eeg_files():
    """

    打开文件选择对话框，允许用户选择一个或多个EEG文件 (.set)。

    Returns:
        list: 用户选择的文件路径列表。如果没有选择文件，则返回空列表。
    """
    # 初始化Tkinter
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    # 打开文件选择对话框并允许多选
    file_paths = askopenfilenames(title="Select EEG Data Files", filetypes=[("EEG Files", "*.set")])

    return list(file_paths)


# 调用函数选择文件
file_paths = select_eeg_files()

# 如果用户没有选择文件，直接退出程序
if not file_paths:
    print("No files selected. Exiting program.")
    exit()

# 初始化保存结果的列表
results = []

# 处理每个数据文件
for data_path in file_paths:
    print(f"Processing file: {data_path}")

    # 读取原始数据
    raw = mne.io.read_raw_eeglab(data_path, preload=True)
    print(raw.ch_names)
    # 设置电极定位
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    raw = raw.resample(250)


    # 送给ICA的需要滤波和平均参考
    raw_filter = raw.copy().filter(l_freq=1.0, h_freq=100.0)
    raw_filter = raw_filter.set_eeg_reference("average")


    # 计算事件和epoch
    events, event_id = mne.events_from_annotations(raw_filter)
    print(f"Event IDs: {event_id}")
    event_ids = {"self": 1, "luxun": 2, "baoyi": 3}

    epochs = mne.Epochs(
        raw_filter, events, event_id=event_ids, tmin=-0.2, tmax=1, baseline=None, detrend=1
    )

    #训练ICA
    ica = ICA(
        n_components=30, max_iter="auto", method="picard",
        random_state=42, fit_params=dict(ortho=False, extended=True)
    )
    ica.fit(epochs)
    epochs.load_data()

    # 自动标记ICA成分
    ic_labels = label_components(epochs, ica, method="iclabel")
    print(f"IC Labels: {ic_labels['labels']},{ic_labels['y_pred_proba']}")
    labels = ic_labels["labels"]
    y_pred_proba = ic_labels["y_pred_proba"]
    exclude_idx = [idx for idx, (label, proba) in enumerate(zip(labels, y_pred_proba)) if
                   label in ["eye blink"] and proba > 0.8]
    excluded_components = len(exclude_idx)


    raw = raw.notch_filter(freqs=(50))
    raw = raw.filter(l_freq=0.1, h_freq=30.0)
    raw = raw.set_eeg_reference('average')

    # 应用ICA于epochs
    epochs_ICAD = mne.Epochs(
        raw, events, event_id=event_ids, tmin=-0.2, tmax=1,
        baseline=None, preload=True, detrend=1
    )
    epochs_ICAD.load_data()

    ica.apply(epochs_ICAD, exclude=exclude_idx)

    # 进行基线校正，并基于绝对电压去除数据
    epochs_ICAD.apply_baseline(baseline=(-0.2, 0))

    # 对ICA后的数据进行自动拒绝
    epochs_ICAD.info['bads'] = []
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 4, 6, 8], random_state=42,
                               n_jobs=-1, verbose=True)
    ar.fit(epochs_ICAD)
    epochs_ICAD_ar, reject_ICAD_log = ar.transform(epochs_ICAD, return_log=True)

    print("自动后试次",len(epochs_ICAD_ar))

    # epochs_ICAD_ar.plot(block=True)
    # epochs_ICAD_ar.drop_bad()
    # print("50uv限制+自动+手动后试次",len(epochs_ICAD_ar))

    reject_criteria = dict(
        eeg=50e-6,
    )  # 150 µV
    epochs_ICAD_ar.drop_bad(reject=reject_criteria)

    print("50uv限制后试次",len(epochs_ICAD_ar))
    self_epochs = epochs_ICAD_ar["self"]
    other_epochs = epochs_ICAD_ar["luxun"]
    baoyi_epochs = epochs_ICAD_ar["baoyi"]

    num_self_epochs = len(self_epochs)
    num_other_epochs = len(other_epochs)
    num_baoyi_epochs = len(baoyi_epochs)
    print("length:", num_self_epochs, num_other_epochs, num_baoyi_epochs)

    trial_counts = {event: round(len(epochs_ICAD_ar[event]) / 60, 2) for event in event_ids}
    # 保存结果到列表
    savepath = r'D:\data\self_exp\CODE\EEG\epochfif\check'
    results.append({
        'file_name': os.path.basename(data_path),
        'trial_counts': trial_counts,
        'excluded_components': excluded_components,
        'excluded_indices': exclude_idx
    })

    output_folder = r"D:\data\self_exp\CODE\EEG\epochfif\check"
    # 生成新的文件名，不包含扩展名，后面加上 '-epo.fif'
    output_file_name = f"{os.path.splitext(os.path.basename(data_path))[0]}-epo.fif"
    # 将目标文件夹路径与生成的文件名拼接在一起，形成最终的完整路径
    output_file = os.path.join(output_folder, output_file_name)

    # # 保存文件
    # output_file = f"{os.path.splitext(os.path.basename(data_path))[0]}-epo.fif"
    epochs_ICAD_ar.save(output_file, overwrite=True)
    print(f"Saved processed data to: {output_file}")

# 将结果保存为CSV文件
df_results = pd.DataFrame(results)
df_results.to_csv('ica_analysis_results.csv', index=False)
print("Results saved to ica_analysis_results.csv")