# Person tracking for HQ-OpenHumanVid

`part_001.tar.gz` ～ `part_012.tar.gz` の MP4 を順に読み、各動画の先頭フレームを Grounding DINO の `person` プロンプトで検出します。検出された各人物の BBOX を SAM 2.1 に渡し、動画全フレームのマスクを保存します。圧縮アーカイブは動画ごとにストリームで読み、part 全体は展開しません。

## 準備

Python 3.10 以上、`uv`、CUDA 対応 GPU が必要です。`uv` で仮想環境を作成し、[`--torch-backend auto`](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection) で GPU ドライバーに合う PyTorch と TorchVision を含む依存関係を導入します。初回実行時には Hugging Face から Grounding DINO と SAM 2.1 の重みがダウンロードされます。

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --torch-backend auto -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available()"
```

## 実行

`--start-part` と `--end-part` は両端を含む 1～12 の範囲です。`--workers` は並列 GPU プロセス数です。GPU は見えているデバイスへ順番に割り当てられます。

```bash
.venv/bin/python track_people.py \
  --dataset /mnt/nas6/shun/VideoUnderstanding/dataset/HQ-OpenHumanVid \
  --start-part 1 --end-part 3 \
  --workers 2 \
  --output /path/to/person-masks
```

複数サーバーでは、例えばサーバー A に `1-4`、B に `5-8`、C に `9-12` を割り当てます。必要に応じて `--gpus 0,1` で使用する GPU を指定できます。`--workers` が GPU 数より多い場合は GPU を共有するため、メモリ使用量に注意してください。処理前の範囲確認には `--dry-run` を使えます。

選択した範囲にアーカイブが一つでも欠けている場合は開始前にエラーになります。

## 出力

各 MP4 に対して以下のディレクトリを作ります。`person_0001` などの番号は先頭フレームで検出した人物 ID で、フレーム間で同じ ID を使います。PNG は元動画の幅・高さで、人物領域が白 (`255`)、背景が黒 (`0`) です。

```text
/path/to/person-masks/
  part_001/
    <clip_id>/
      metadata.json
      person_0001/
        000000.png
        000001.png
        ...
      person_0002/
        000000.png
        ...
```

`metadata.json` には入力アーカイブ内のパス、フレーム数、画像サイズ、各人物の初期 BBOX (`xyxy`) と検出スコアを記録します。先頭フレームに人物がいない動画には `status: "no_person"` の `metadata.json` だけを保存します。完了済み動画は再実行時にスキップします。失敗した動画は標準出力に表示し、最後に非ゼロの終了コードを返します。

この処理は先頭フレームで検出された人物を追跡します。途中から登場する人物は追加検出しません。
