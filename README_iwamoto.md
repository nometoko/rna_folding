# 岩本の作業ログ

## 2025/04/06

### [host の Discussion](https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/565064) を見て

ホストの言う戦略

- [RibonanzaNet](https://www.biorxiv.org/content/10.1101/2024.02.24.581671v2)を Fine-tune する。
- マルチプルアラインメントの情報を使う。
- 文献知識の活用。
- 複数の構造を持つ RNA を考慮する。
  この時、seed の違いで異なる構造を得ることは考えない。
- 仮の構造を生成し、こそから更に推論を行う。
- 合成 RNA の構造を augmentation に用いる。

### [RibonanzaNet Fine-tune の Discussion](https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/565306) を見て

#### Fine-tune の loss

0. `calculate_distance_matrix`関数

   **1 つの構造の中の**各塩基同士の距離を計算

   ```python
   def calculate_distance_matrix(X,Y,epsilon=1e-4):
       # X[:,None]はXの各残基を行ベクトルに、Y[None,:]はYの各残基を列ベクトルに変換したもの
       # X[:,None]-Y[None,:]はXの各残基とYの各残基の距離を計算した len(X) x len(Y) 行列
       return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()
   ```

1. `dRMSD` (Root Mean Square Deviation)

   **通常、`pred_x`と`pred_y`, `gt_x`と`gt_y`は同じ構造を持つ。**

   ```python
   def dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4,Z=10,d_clamp=None):
       '''
       pred_x: 予測された構造
       pred_y: 予測された構造
       gt_x: 実際の構造
       gt_y: 実際の構造
       '''

       # pred_xとpred_yの各残基の座標を使って距離行列を計算
       pred_dm=calculate_distance_matrix(pred_x,pred_y)
       # gt_xとgt_yの各残基の座標を使って距離行列を計算
       gt_dm=calculate_distance_matrix(gt_x,gt_y)

       # gt_dmのNaNを除外するマスクを作成
       mask=~torch.isnan(gt_dm)
       # 対角成分(自分自身との距離)を除外するマスクを作成
       mask[torch.eye(mask.shape[0]).bool()]=False

       # d_clampが指定されている場合、距離の差をd_clampでクリップする
       if d_clamp is not None:
           # pred_dmとgt_dmの距離の差を計算
           # clipで指定された範囲 (0 <= loss <= d_clamp**2) に収める
           rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
       else:
           rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

       return rmsd.sqrt().mean()/Z
   ```

2. `local_dRMSD` (Local Root Mean Square Deviation)

   `dRMSD`とは mask の作り方が異なり、**d_clamp より大きい距離は loss の計算から除外される**。

   ```python
   def local_dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4,Z=10,d_clamp=30):
       # 上と同じ
       pred_dm=calculate_distance_matrix(pred_x,pred_y)
       gt_dm=calculate_distance_matrix(gt_x,gt_y)

       # gt_dmがNaN　+ dt_dmがd_clamp以上の距離を除外するマスクを作成
       mask=(~torch.isnan(gt_dm)) * (gt_dm < d_clamp)
       mask[torch.eye(mask.shape[0]).bool()]=False

       rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

       return rmsd.sqrt().mean()/Z
   ```

3. `dMAE` (Mean Absolute Error)

> [!NOTE]
> notebook には`dRMAE`と書いてあるが、実装では root を取っていないので`dMAE`の誤植と思われる。

```python
def dMAE(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)

    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    # dRMSDでは2乗していたが、dRMAEでは絶対値を取る
    # d_clampの処理をしないのは2乗しないため大きい値を取らないから?
    rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])

    # rootも取らない
    return rmsd.mean()/Z
```

4. `align_svd_mae`

   特異値分解を用いて、2 つの構造のアライメントを行い、MAE loss を計算する。

   ```python
   def align_svd_mae(input, target, Z=10):

       assert input.shape == target.shape, "Input and target must have the same shape"

       # sum(-1)で各塩基の座標を足し合わせて、NaNを含む行(塩基)を除外するマスクを作成
       mask=~torch.isnan(target.sum(-1))

       input=input[mask]
       target=target[mask]

       # x, y, zの座標を持つ点群の重心を計算
       # 結果として1x3のテンソル(x, y, zの平均を持つ)が得られる
       centroid_input = input.mean(dim=0, keepdim=True)
       centroid_target = target.mean(dim=0, keepdim=True)

       # 各塩基の座標から重心を引いて、中心化する (重心を原点に移動)
       # detach()は重心に関して、勾配の計算を行わないようにするため (誤差逆伝播の計算を行わない)
       input_centered = input - centroid_input.detach()
       target_centered = target - centroid_target

       # Procrustes analysis
       # 2つのdatasetが最もマッチするような回転を求める

       cov_matrix = input_centered.T @ target_centered

       # SVD to find optimal rotation
       U, S, Vt = torch.svd(cov_matrix)

       # Compute rotation matrix
       R = Vt @ U.T

       # torch.det(R)が負の場合、回転ではなく反転が起こっているので、det(R)を正にするためにVtの最後の行を反転させる
       if torch.det(R) < 0:
           Vt[-1, :] *= -1
           R = Vt @ U.T

       # Rotate input
       aligned_input = (input_centered @ R.T.detach()) + centroid_target.detach()

       # Calculate MAE loss
       return torch.abs(aligned_input-target).mean()/Z
   ```

#### Fine-tune の train

- optimizer

  - Adam
  - param: `learning rate`: 1e-4
  - param: `weight decay`: 0.0 \
     L2 正則化なし \
     Alfa Fold に従っている

- scheduler

  - CosineAnnealingLR \
     学習率をコサイン関数に従って減少させる。 \
     `cos_epoch`で指定した epoch 数から徐々に減少させる。

    - param: `T_max` \
       `iteration`の最大値である。つまり、**学習率が 0 に達する epoch 数**であり、以下のように指定されている。 \
       `(epoch - cos_epoch) * len(train_loader) // batch_size`

      - `epoch - cos_epoch` \
         `全体のepoch数 - cosineAnnealingを開始するepoch`であり、学習率を減少させる epoch 数を表す。
      - `len(train_loader) // batch_size` \
         `train_loader`の長さを`batch_size`で割った値であり、1 epoch あたりのイテレーション数を表す。 \
         つまり、1 epoch あたりのイテレーション数は、`train_loader`の長さを`batch_size`で割った値である。

> [!NOTE]
> ChatGPT によると、`len(train_loader)`自体が iteration 数を表すので、`batch_size`で割る必要はないとのこと。 \
> 真偽のほどは不明。

      結果として、**cosineAnnealing を適用する epoch 数 \* 1 epoch あたりの iteration 数 = cosineAnnealing を適用する iteration 数**が`T_max`となる。
