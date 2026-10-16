# 統合実行パッケージ

研究上のL1/L2/L3の区分は内部に残し、実行入口は keiba_runtime に統一する。
既存のレースIDパイプラインを一回だけ呼び、出力は既存のモバイル向け
KEIBA_GITHUB_RACE_OUTPUT_BATCH_V1 JSONから変更しない。

実行:

    python -m keiba_runtime --race 202507010212 --manifest /path/to/github-manifest.json --output /path/to/mobile.json

複数レースは --race を繰り返す。

内部: オッズ未補正AI順位、元のANCHOR_FULL確率×同時点オッズに基づく
期待値AI順位、凍結L2.1 STANDARD 50条件、正式L3 V3
REVERSE_P3_RACE_MAX300、同日資金不足の停止まで、既存コードを変更せず実行。

注意: GitHub側で真正な特徴量、発走前オッズ、元のL1.7 p3、予測モデル
入力を自動生成する上流は未完成。実データなしでは BLOCKED であり、
レースIDだけから実予測できると主張しない。学習済みモデル、研究順位、
結果、FINALオッズの代理投入はしない。PAPER_ONLY を READY と見なさない。
本件は実行入口の統合のみで、凍結仕様・入力保存方式・main は変更しない。
