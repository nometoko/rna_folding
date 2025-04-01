## Environment Setting

### Installation

1. Install Anaconda to your linux system.

> [!NOTE]
> **To Sugawara** \
> 念の為一連の操作を実行する前に、`.zshrc`に書かれた anaconda に関する設定を消しておくこと

   ```
   % curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
   % bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh
   ```

   a. まず、`Enter key` を押す。 \
   b. その後`more`が起動するので、`space key`で最後まで行った後、`yes`を入力。 \
   c. Anaconda の install 先を聞かれるので、デフォルトのまま`Enter key`を押す。
   (その場合、`${HOME}/anaconda3`にインストールされる) \
   d. `.zshrc`を変更するか聞かれるので、`yes`を入力。 \
   e. `conda --version`を実行して、`24.x.x`と表示されれば成功。

2. Create a conda environment
   ```
   % conda create -n rna_folding python=3.9
   % conda activate rna_folding
   % which python # Python 3.9.xであればOK
   Python 3.9.21
   % conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge
   % conda install 'numpy=1.*'
   % python
   >>> import torch
   >>> print(torch.cuda.is_available()) # CUDAが使えるか確認
   True # CUDAが使える場合
   >>> print(torch.cuda.current_device()) # CUDAのデバイス番号を確認
   0 # CUDAのデバイス番号を確認
   >>> print(torch.cuda.get_device_name(0)) # CUDAのデバイス名を確認
   Tesla V100-PCIE-32GB # CUDAのデバイス名を確認
   >>> print(torch.cuda.device_count()) # CUDAのデバイス数を確認
   4 # CUDAのデバイス数を確認
   ```
### Share Environment

Anacondaの場合、`environment.yaml`を介してパッケージのバージョンを指定して環境を共有することができる。

1. `environment.yaml`を作成する
   ```
   % conda env export > environment.yaml
   % nvim environment.yaml #最終行の`prefix`を消す
   ```

> [!NOTE]
> `prefix`の部分は絶対パスを含むため、削除する

2. `environment.yaml`を使って環境を作成する
   ```
   % conda env create -f environment.yaml
   % conda activate rna_folding
   ```

3. `environment.yaml`を使って環境を更新する
   ```
   % conda env update --file environment.yaml
   ```
