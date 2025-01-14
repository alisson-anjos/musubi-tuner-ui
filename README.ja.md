# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md)

## 目次

- [はじめに](#はじめに)
    - [最近の更新](#最近の更新)
    - [リリースについて](#リリースについて)
- [概要](#概要)
    - [ハードウェア要件](#ハードウェア要件)
    - [特徴](#特徴)
- [インストール](#インストール)
- [モデルのダウンロード](#モデルのダウンロード)
    - [HunyuanVideoの公式モデルを使う](#HunyuanVideoの公式モデルを使う)
    - [Text EncoderにComfyUI提供のモデルを使う](#Text-EncoderにComfyUI提供のモデルを使う)
- [使い方](#使い方)
    - [データセット設定](#データセット設定)
    - [latentの事前キャッシュ](#latentの事前キャッシュ)
    - [Text Encoder出力の事前キャッシュ](#Text-Encoder出力の事前キャッシュ)
    - [学習](#学習)
    - [推論](#推論)
    - [LoRAの形式の変換](#LoRAの形式の変換)
- [その他](#その他)
    - [SageAttentionのインストール方法](#SageAttentionのインストール方法)
- [免責事項](#免責事項)
- [コントリビューションについて](#コントリビューションについて)
- [ライセンス](#ライセンス)

## はじめに

このリポジトリは、HunyuanVideoのLoRA学習用のコマンドラインツールです。このリポジトリは非公式であり、公式のHunyuanVideoリポジトリとは関係ありません。

*リポジトリは開発中です。*

### 最近の更新

- 2025/01/14
    - `hv_generate_video.py`に、LoRAマージ後のDiTモデルを保存する`--save_merged_model`オプションを暫定的に追加しました。詳細は[推論](#推論)を参照してください。

- 2025/01/13
    - 学習中のサンプル画像（動画）がぼやける現象に対応するため、サンプル生成時の設定を変更しました。詳細は[こちら](./docs/sampling_during_training.md)をご参照ください。
        - 推論時にdiscrete flow shiftとguidance scaleを正しく設定する必要がありますが、学習時の設定がそのまま使われていたため、この事象が発生していました。デフォルト値を設定したため、改善されると思われます。また`--fs`でdiscrete flow shiftを、`--g`でguidance scaleを指定できます。

- 2025/01/12
    - 学習中のサンプル画像生成が可能になりました。NSFW-API氏に感謝いたします。詳細は[こちらのドキュメント](./docs/sampling_during_training.md)を参照してください。
    - データセットごとに繰り返し回数を指定できるようになりました。指定した数だけデータセットを繰り返し、1 epochとして学習します。`.toml`に`num_repeats`を指定してください。詳細は[こちらのドキュメント](./dataset/dataset_config.md)を参照してください。
    - LoRAの適用対象モジュールから、double blocksの`img_mod`および`txt_mod`、single blocksの`modulation`をデフォルトで除外するようにしました。コミュニティからの報告によると、これにより学習結果が改善されるようです。`--network_args`で`exclude_patterns`および`include_patterns`を指定して、適用対象モジュールを変更できます。詳細は[こちらのドキュメント](./docs/advanced_config.md)を参照してください。
        - 以前の重みを`--network_weights`で指定して学習を再開する場合、お手数ですが `--network_args "include_patterns=[r'.*(img_mod|txt_mod|modulation).*']"`を指定してください。
    - LoRA+についてもそちらのドキュメントに追加しました。

- 2025/01/11
    - LoRAのメタデータに保存されていた、学習対象モデル（DiT、VAE）のハッシュ値を削除しました。ハッシュ値はほぼ使用されておらず、計算に時間が掛かるためです。もし問題があればご連絡ください。
    - fp8に変換したDiTモデルの重みを[こちら](https://huggingface.co/kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial)で公開しました。`--fp8_base`指定時のみ使用可能です。ダウンロードして、`--dit`に`mp_rank_00_model_states_fp8.safetensors`のフルパスを指定してください。学習開始時の初期化処理が速くなります。

- 2025/01/10
    - `--split_attn`を指定しない場合でも、`--split_attn`が適用された状態になっていた不具合を修正しました。
    - 学習および推論でxformersが利用可能になりました。xformers利用時は`--split_attn`の指定が必要です。
    - `flash`モードでの不具合を修正し、動作を確認しました。また推論時に、`--split_attn`を指定可能にしました。
    - 学習時の簡易な速度比較を行うと、1280x720解像度の画像による学習、batch size=3、RTX 4090、Windows 10の環境では、`flash`  > `flash` + `--split_attn` == `xformers` + `--split_attn` > `sdpa` > `sdpa` + `--split_attn` となりました（`flash`が最も高速）。
        - `sdpa`は`flash`より12%程度遅く、`xformers` + `--split_attn`は`flash`より4%程度遅いようです。
    - 推論時は `flash` >= `sageattn` > `xformers` > `sdpa` となりました（いずれも`--split_attn`を指定）。ただし、`sdpa`と`flash`の速度差は10%程度です。またメモリ使用量はほぼ同じでした。

- 2025/01/08
    - __重要な更新__：latentsがキャッシュ時と学習時の二回、scalingされる不具合を修正しました。お手数ですが、cache_latents.pyを（`--skip_existing`を指定せずに）再度実行して、latentsを再キャッシュしてください。

### リリースについて

Musubi Tunerの解説記事執筆や、関連ツールの開発に取り組んでくださる方々に感謝いたします。このプロジェクトは開発中のため、互換性のない変更や機能追加が起きる可能性があります。想定外の互換性問題を避けるため、参照用として[リリース](https://github.com/kohya-ss/musubi-tuner/releases)をお使いください。

最新のリリースとバージョン履歴は[リリースページ](https://github.com/kohya-ss/musubi-tuner/releases)で確認できます。

## 概要

### ハードウェア要件

- VRAM: 静止画での学習は12GB以上推奨、動画での学習は24GB以上推奨。
    - *解像度等の学習設定により異なります。*12GBでは解像度 960x544 以下とし、`--blocks_to_swap`、`--fp8_llm`等の省メモリオプションを使用してください。
- メインメモリ: 64GB以上を推奨、32GB+スワップで動作するかもしれませんが、未検証です。

### 特徴

- 省メモリに特化
- Windows対応（Linuxでの動作報告もあります）
- マルチGPUには対応していません

## インストール

Python 3.10以上を使用してください（3.10で動作確認済み）。

適当な仮想環境を作成し、ご利用のCUDAバージョンに合わせたPyTorchとtorchvisionをインストールしてください。

PyTorchはバージョン2.5.1以上を使用してください（[補足](#PyTorchのバージョンについて)）。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

以下のコマンドを使用して、必要な依存関係をインストールします。

```bash
pip install -r requirements.txt
```

オプションとして、FlashAttention、SageAttention（推論にのみ使用、インストール方法は[こちら](#SageAttentionのインストール方法)を参照）を使用できます。

また、`ascii-magic`（データセットの確認に使用）、`matplotlib`（timestepsの可視化に使用）、`tensorboard`（学習ログの記録に使用）を必要に応じてインストールしてください。

```bash
pip install ascii-magic matplotlib tensorboard
```

## モデルのダウンロード

以下のいずれかの方法で、モデルをダウンロードしてください。

### HunyuanVideoの公式モデルを使う 

[公式のREADME](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md)を参考にダウンロードし、任意のディレクトリに以下のように配置します。

```
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

### Text EncoderにComfyUI提供のモデルを使う

こちらの方法の方がより簡単です。DiTとVAEのモデルはHumyuanVideoのものを使用します。

https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/transformers から、[mp_rank_00_model_states.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt) をダウンロードし、任意のディレクトリに配置します。

（同じページにfp8のモデルもありますが、未検証です。）

`--fp8_base`を指定して学習する場合は、`mp_rank_00_model_states.pt`の代わりに、[こちら](https://huggingface.co/kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial)の`mp_rank_00_model_states_fp8.safetensors`を使用可能です。（このファイルは非公式のもので、重みを単純にfloat8_e4m3fnに変換したものです。）

また、https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/vae から、[pytorch_model.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt) をダウンロードし、任意のディレクトリに配置します。

Text EncoderにはComfyUI提供のモデルを使用させていただきます。[ComyUIのページ](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)を参考に、https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/text_encoders から、llava_llama3_fp16.safetensors （Text Encoder 1、LLM）と、clip_l.safetensors （Text Encoder 2、CLIP）をダウンロードし、任意のディレクトリに配置します。

（同じページにfp8のLLMモデルもありますが、動作未検証です。）

## 使い方

### データセット設定

[こちら](./dataset/dataset_config.md)を参照してください。

### latentの事前キャッシュ

latentの事前キャッシュは必須です。以下のコマンドを使用して、事前キャッシュを作成してください。

```bash
python cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

その他のオプションは`python cache_latents.py --help`で確認できます。

VRAMが足りない場合は、`--vae_spatial_tile_sample_min_size`を128程度に減らし、`--batch_size`を小さくしてください。

`--debug_mode image` を指定するとデータセットの画像とキャプションが新規ウィンドウに表示されます。`--debug_mode console`でコンソールに表示されます（`ascii-magic`が必要）。

### Text Encoder出力の事前キャッシュ

Text Encoder出力の事前キャッシュは必須です。以下のコマンドを使用して、事前キャッシュを作成してください。

```bash
python cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

その他のオプションは`python cache_text_encoder_outputs.py --help`で確認できます。

`--batch_size`はVRAMに合わせて調整してください。

VRAMが足りない場合（16GB程度未満の場合）は、`--fp8_llm`を指定して、fp8でLLMを実行してください。

### 学習

以下のコマンドを使用して、学習を開始します（実際には一行で入力してください）。

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 1e-3 --gradient_checkpointing 
     --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling sigmoid --discrete_flow_shift 1.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

その他のオプションは`python hv_train_network.py --help`で確認できます（ただし多くのオプションは動作未確認です）。

`--fp8_base`を指定すると、DiTがfp8で学習されます。未指定時はmixed precisionのデータ型が使用されます。fp8は大きく消費メモリを削減できますが、品質は低下する可能性があります。`--fp8_base`を指定しない場合はVRAM 24GB以上を推奨します。また必要に応じて`--blocks_to_swap`を使用してください。

VRAMが足りない場合は、`--blocks_to_swap`を指定して、一部のブロックをCPUにオフロードしてください。最大36が指定できます。

（block swapのアイデアは2kpr氏の実装に基づくものです。2kpr氏にあらためて感謝します。）

`--sdpa`でPyTorchのscaled dot product attentionを使用します。`--flash_attn`で[FlashAttention]:(https://github.com/Dao-AILab/flash-attention)を使用します。`--xformers`でxformersの利用も可能ですが、xformersを使う場合は`--split_attn`を指定してください。`--sage_attn`でSageAttentionを使用しますが、SageAttentionは現時点では学習に未対応のため、正しく動作しません。

`--split_attn`を指定すると、attentionを分割して処理します。速度が多少低下しますが、VRAM使用量はわずかに減ります。

サンプル動画生成は現時点では実装されていません。

学習されるLoRAの形式は、`sd-scripts`と同じです。

`--show_timesteps`に`image`（`matplotlib`が必要）または`console`を指定すると、学習時のtimestepsの分布とtimestepsごとのloss weightingが確認できます。

学習中のサンプル画像生成については、[こちらのドキュメント](./docs/sampling_during_training.md)を参照してください。その他の高度な設定については[こちらのドキュメント](./docs/advanced_config.md)を参照してください。

適切な学習率、学習ステップ数、timestepsの分布、loss weightingなどのパラメータは、現時点ではわかっていません。情報提供をお待ちしています。

### 推論

以下のコマンドを使用して動画を生成します。

```bash
python hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

その他のオプションは`python hv_generate_video.py --help`で確認できます。

`--fp8`を指定すると、DiTがfp8で推論されます。fp8は大きく消費メモリを削減できますが、品質は低下する可能性があります。

VRAMが足りない場合は、`--blocks_to_swap`を指定して、一部のブロックをCPUにオフロードしてください。最大38が指定できます。

`--attn_mode`には`flash`、`torch`、`sageattn`、`xformers`または`sdpa`（`torch`指定時と同じ）のいずれかを指定してください。それぞれFlashAttention、scaled dot product attention、SageAttention、xformersに対応します。デフォルトは`torch`です。SageAttentionはVRAMの削減に有効です。

`--split_attn`を指定すると、attentionを分割して処理します。SageAttention利用時で10%程度の高速化が見込まれます。

`--output_type`には`both`、`latent`、`video`、`images`のいずれかを指定してください。`both`はlatentと動画の両方を出力します。VAEでOut of Memoryエラーが発生する場合に備えて、`both`を指定することをお勧めします。`--latent_path`に保存されたlatentを指定し、`--output_type video` （または`images`）としてスクリプトを実行すると、VAEのdecodeのみを行えます。

`--seed`は省略可能です。指定しない場合はランダムなシードが使用されます。

`--video_length`は「4の倍数+1」を指定してください。

`--flow_shift`にタイムステップのシフト値（discrete flow shift）を指定可能です。省略時のデフォルト値は7.0で、これは推論ステップ数が50の時の推奨値です。HunyuanVideoの論文では、ステップ数50の場合は7.0、ステップ数20未満（10など）で17.0が推奨されています。

`--save_merged_model`オプションで、LoRAマージ後のDiTモデルを保存できます。`--save_merged_model path/to/merged_model.safetensors`のように指定してください。なおこのオプションを指定すると推論は行われません。

### LoRAの形式の変換

ComfyUIで使用可能な形式（Diffusion-pipeと思われる）への変換は以下のコマンドで行えます。

```bash
python convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

`--input`と`--output`はそれぞれ入力と出力のファイルパスを指定してください。

`--target`には`other`を指定してください。`default`を指定すると、他の形式から当リポジトリの形式に変換できます。

## その他

### SageAttentionのインストール方法

sdbds氏によるWindows対応のSageAttentionのwheelが https://github.com/sdbds/SageAttention-for-windows で公開されています。triton をインストールし、Python、PyTorch、CUDAのバージョンが一致する場合は、[Releases](https://github.com/sdbds/SageAttention-for-windows/releases)からビルド済みwheelをダウンロードしてインストールすることが可能です。sdbds氏に感謝します。

参考までに、以下は、SageAttentionをビルドしインストールするための簡単な手順です。Microsoft Visual C++ 再頒布可能パッケージを最新にする必要があるかもしれません。

1. Pythonのバージョンに応じたtriton 3.1.0のwhellを[こちら](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5)からダウンロードしてインストールします。

2. Microsoft Visual Studio 2022かBuild Tools for Visual Studio 2022を、C++のビルドができるよう設定し、インストールします。（上のRedditの投稿を参照してください）。

3. 任意のフォルダにSageAttentionのリポジトリをクローンします。
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

    なお `git clone https://github.com/sdbds/SageAttention-for-windows.git` で、前述のsdbds氏のリポジトリを使用することで、手順4.を省略できます。

4. `SageAttention/csrc`フォルダ内の`math.cuh`を開き、71行目と146行目の `ushort` を `unsigned short` に変更して保存します。

5. スタートメニューから Visual Studio 2022 内の `x64 Native Tools Command Prompt for VS 2022` を選択してコマンドプロンプトを開きます。

6. venvを有効にし、SageAttentionのフォルダに移動して以下のコマンドを実行します。DISTUTILSが設定されていない、のようなエラーが出た場合は `set DISTUTILS_USE_SDK=1`としてから再度実行してください。
    ```shell
    python setup.py install
    ```

以上でSageAttentionのインストールが完了です。

### PyTorchのバージョンについて

`--attn_mode`に`torch`を指定する場合、2.5.1以降のPyTorchを使用してください（それより前のバージョンでは生成される動画が真っ黒になるようです）。

古いバージョンを使う場合、xformersやSageAttentionを使用してください。

## 免責事項

このリポジトリは非公式であり、公式のHunyuanVideoリポジトリとは関係ありません。また、このリポジトリは開発中で、実験的なものです。テストおよびフィードバックを歓迎しますが、以下の点にご注意ください：

- 実際の稼働環境での動作を意図したものではありません
- 機能やAPIは予告なく変更されることがあります
- いくつもの機能が未検証です
- 動画学習機能はまだ開発中です

問題やバグについては、以下の情報とともにIssueを作成してください：

- 問題の詳細な説明
- 再現手順
- 環境の詳細（OS、GPU、VRAM、Pythonバージョンなど）
- 関連するエラーメッセージやログ

## コントリビューションについて

コントリビューションを歓迎します。ただし、以下にご注意ください：

- メンテナーのリソースが限られているため、PRのレビューやマージには時間がかかる場合があります
- 大きな変更に取り組む前には、議論のためのIssueを作成してください
- PRに関して：
    - 変更は焦点を絞り、適度なサイズにしてください
    - 明確な説明をお願いします
    - 既存のコードスタイルに従ってください
    - ドキュメントが更新されていることを確認してください

## ライセンス

`hunyuan_model`ディレクトリ以下のコードは、[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)のコードを一部改変して使用しているため、そちらのライセンスに従います。

他のコードはApache License 2.0に従います。一部Diffusersのコードをコピー、改変して使用しています。
