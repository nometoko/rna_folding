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
