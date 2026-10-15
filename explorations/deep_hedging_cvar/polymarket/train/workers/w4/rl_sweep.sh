#!/bin/bash
# w4 step 4: BC from the learned quoter (one BC student per data-seed model) then reward-mode-2 PPO
# with a decaying KL to the BC student. 8 configs x 3 seeds, 3 runs in parallel, 1 thread each.
#   bash rl_sweep.sh bc        # 3 BC students (ckpt/bc_m{0,1,2}.pt)
#   bash rl_sweep.sh rl        # the 24 PPO runs (skips finished ones)
set -u
W4="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$(cd "$W4/../.." && pwd)"
PM="$(cd "$TRAIN/.." && pwd)"
V2="$PM/data/build_v2"
source "$PM/../.venv/bin/activate"
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive
JOBS=${JOBS:-3}
STEPS=${STEPS:-30e6}
QJSON=${QJSON:-$W4/runs/q1_best.json}

if [ "$1" = bc ]; then
  for m in 0 1 2; do
    python "$W4/bc_w4.py" --quoter "$QJSON" --model-idx $m -- --num-envs 1024 --steps 2048 --epochs 8 \
      --hidden 64 --layers 2 --value --gamma 0.9988 --seed $((m + 1)) --threads 1 \
      --env-kw data_dir=$V2 reward_mode=2 markout_h=60 mark_mode=1 \
      --out "$W4/ckpt/bc_m$m.pt" > "$W4/runs/bc_m$m.log" 2>&1 &
  done
  wait
  exit 0
fi

# name | extra train_ppo args
CONFIGS=(
  "c1_h60_kl0.1_d0.5|--env-kw data_dir=$V2 reward_mode=2 markout_h=60 mark_mode=1 --kl-coef 0.1 --kl-decay 0.5"
  "c2_h120_kl0.1_d0.5|--env-kw data_dir=$V2 reward_mode=2 markout_h=120 mark_mode=1 --kl-coef 0.1 --kl-decay 0.5"
  "c3_h60_kl0.3_d0.5|--env-kw data_dir=$V2 reward_mode=2 markout_h=60 mark_mode=1 --kl-coef 0.3 --kl-decay 0.5"
  "c4_h60_kl0.03_d0.25|--env-kw data_dir=$V2 reward_mode=2 markout_h=60 mark_mode=1 --kl-coef 0.03 --kl-decay 0.25"
  "c5_h60_kl0.1_d0.9|--env-kw data_dir=$V2 reward_mode=2 markout_h=60 mark_mode=1 --kl-coef 0.1 --kl-decay 0.9"
  "c6_h60_kl0.1_d0.5_lr1e-4|--env-kw data_dir=$V2 reward_mode=2 markout_h=60 mark_mode=1 --kl-coef 0.1 --kl-decay 0.5 --lr 1e-4"
  "c7_h60_kl1_d0.5|--env-kw data_dir=$V2 reward_mode=2 markout_h=60 mark_mode=1 --kl-coef 1.0 --kl-decay 0.5"
  "c8_h60_kl0.1_d0.5_inv1e-3|--env-kw data_dir=$V2 reward_mode=2 markout_h=60 mark_mode=1 --kl-coef 0.1 --kl-decay 0.5 --inv-penalty 1e-3"
)
[ -n "${ONLY:-}" ] && CONFIGS=("${CONFIGS[@]:0:$ONLY}")

run_one() {
  local name="$1" extra="$2" s="$3"
  local out="$W4/runs/rl/${name}_s$s"
  [ -f "$out/result.json" ] && return 0
  mkdir -p "$out"
  eval python "$TRAIN/train_ppo.py" --config "$W4/r7_base_cfg.json" --env real --init "$W4/ckpt/bc_m$((s - 1)).pt" \
    --seed $s --timesteps $STEPS --threads 1 --no-train-eval --variants central_1s pess_1s central_2s \
    --ledger "$W4/trials.jsonl" --tag "w4_rl_${name}" --out "$out" --quiet $extra > "$out/stdout.log" 2>&1
}

for s in 1 2 3; do
  for c in "${CONFIGS[@]}"; do
    name="${c%%|*}"; extra="${c#*|}"
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    run_one "$name" "$extra" "$s" &
  done
done
wait
