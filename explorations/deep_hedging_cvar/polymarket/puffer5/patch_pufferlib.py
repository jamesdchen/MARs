"""Two small, idempotent edits to a PufferLib 5.0 checkout for the MM env.

    python patch_pufferlib.py /content/PufferLib     # apply
    python patch_pufferlib.py --selftest [PUFFERLIB]  # CPU check, no CUDA needed

1. src/pufferl.cu, run_train: before the final evaluation of a run (the
   score a train run reports, and the one point a sweep trial gives Protein
   with sweep.downsample = 1), apply the [eval_env] section of the ini as
   [env] overrides, and restore [env] afterwards (so the run's log ini still
   records the training split). polymarket_mm.ini sets [eval_env] split = 1:
   training runs on the train split, the score comes from validation. The
   patched trainer prints "eval_env: final eval with [env] split = 1 (was 0)"
   before that evaluation; the notebook checks for it in every log.
   Without the patch, [eval_env] is ignored and the score is a train-split
   number, so the notebook refuses to sweep on an unpatched build.
2. config/default.ini: drop the generic [sweep.*] blocks (they sweep
   vec.total_agents, num_layers up to 8, vf_clip_coef ...), so sweep.ini is
   the whole search space. The [sweep] section itself stays.

Pinned commits: 6ffa5b1 and ea7f989 (identical run_train).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = 'polymarket_mm patch'

FUNCS = r'''
// polymarket_mm patch (polymarket/puffer5/patch_pufferlib.py): [eval_env]
// holds [env] overrides for the final evaluation of a train run, so its score
// (the sweep metric) comes from another data split. Returns the old values.
static char** eval_env_apply(Ini* ini) {
    Dict* over = puf_ini_section(ini, "eval_env", 1);
    Dict* env = puf_ini_section(ini, "env", 0);
    char** saved = (char**)calloc(over->size + 1, sizeof(char*));
    for (int i = 0; i < over->size; i++) {
        DictItem* item = dict_find(env, over->items[i].key);
        if (!item || !item->str || !over->items[i].str) {
            fprintf(stderr, "[eval_env] %s: not a string key of [env]\n", over->items[i].key);
            exit(1);
        }
        saved[i] = dict_strdup(item->str);
        puf_ini_set(env, over->items[i].key, over->items[i].str);
        printf("eval_env: final eval with [env] %s = %s (was %s)\n",
            over->items[i].key, over->items[i].str, saved[i]);
    }
    fflush(stdout);
    return saved;
}

static void eval_env_restore(Ini* ini, char** saved) {
    Dict* over = puf_ini_section(ini, "eval_env", 1);
    Dict* env = puf_ini_section(ini, "env", 0);
    for (int i = 0; i < over->size; i++) {
        puf_ini_set(env, over->items[i].key, saved[i]);
        free(saved[i]);
    }
    free(saved);
}
'''

FUNC_ANCHOR = 'TrainResult run_train(Ini* ini, TrainContext* ctx) {'
EVAL_OLD = '''        PuffeRL* ep = eval_make(ini, &eval_ctx, EVAL_SCORE, 0);
        EvalResult r = eval_loop(ini, ep, EVAL_SCORE, 1, 0, eval_episodes,
            &last_log, eval_epoch);
        result.score = result.scores[result.points - 1] = r.score;
        close_pufferl(ep);
'''
EVAL_NEW = f'''        char** eval_env_saved = eval_env_apply(ini);  // {MARK}
        PuffeRL* ep = eval_make(ini, &eval_ctx, EVAL_SCORE, 0);
        EvalResult r = eval_loop(ini, ep, EVAL_SCORE, 1, 0, eval_episodes,
            &last_log, eval_epoch);
        result.score = result.scores[result.points - 1] = r.score;
        close_pufferl(ep);
        eval_env_restore(ini, eval_env_saved);  // {MARK}
'''


def patch_source(text):
    if MARK in text:
        return text, False
    for anchor in (FUNC_ANCHOR, EVAL_OLD):
        if text.count(anchor) != 1:
            raise SystemExit(f'pufferl.cu: expected one occurrence of {anchor[:60]!r}; '
                             'not the pinned PufferLib 5.0?')
    text = text.replace(FUNC_ANCHOR, FUNCS.lstrip('\n') + '\n' + FUNC_ANCHOR)
    return text.replace(EVAL_OLD, EVAL_NEW), True


def strip_sweep_blocks(text):
    blocks = re.split(r'(?m)^(?=\[)', text)
    kept = [b for b in blocks if not b.startswith('[sweep.')]
    return ''.join(kept), len(blocks) - len(kept)


def apply(root):
    src = os.path.join(root, 'src', 'pufferl.cu')
    with open(src) as f:
        text, changed = patch_source(f.read())
    if changed:
        with open(src, 'w') as f:
            f.write(text)
    ini = os.path.join(root, 'config', 'default.ini')
    with open(ini) as f:
        text, n = strip_sweep_blocks(f.read())
    with open(ini, 'w') as f:
        f.write(text)
    print(f'{src}: {"patched" if changed else "already patched"}; '
          f'{ini}: removed {n} [sweep.*] blocks')


def is_patched(root):
    with open(os.path.join(root, 'src', 'pufferl.cu')) as f:
        return MARK in f.read()


HARNESS = r'''
#include "ini.h"
#include <assert.h>
%s
static const char* env_str(Ini* ini, const char* key) {
    return dict_get_str(puf_ini_section(ini, "env", 0), key);
}
int main(int argc, char** argv) {
    // as the trainer: default.ini, then the env ini, then argv overrides
    Ini ini = {0};
    puf_ini_load_file(&ini, argv[1]);
    puf_ini_load_file(&ini, argv[2]);
    for (int i = 3; i < argc; i++) {
        puf_ini_apply_arg(&ini, "base", argv[i], i);
    }
    char before[256];
    snprintf(before, sizeof(before), "%%s", env_str(&ini, "split"));
    double v0 = dict_get(puf_ini_section(&ini, "env", 0), "split");
    char** saved = eval_env_apply(&ini);
    printf("during eval: split=%%s (%%g) data_dir=%%s\n", env_str(&ini, "split"),
        dict_get(puf_ini_section(&ini, "env", 0), "split"), env_str(&ini, "data_dir"));
    double v1 = dict_get(puf_ini_section(&ini, "env", 0), "split");
    eval_env_restore(&ini, saved);
    printf("after: split=%%s\n", env_str(&ini, "split"));
    assert(strcmp(env_str(&ini, "split"), before) == 0);
    assert(dict_get(puf_ini_section(&ini, "env", 0), "split") == v0 && v1 == 1.0 && v0 == 0.0);
    puf_ini_free(&ini);
    return 0;
}
'''


def selftest(root):
    """Compile the inserted functions against src/ini.h and run them on
    polymarket_mm.ini (+ sweep.ini, as the sweep loads it), directly and with
    the argv a sweep trial gets; then patch copies of pufferl.cu/default.ini."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, 'polymarket_mm.ini')
        with open(cfg, 'w') as f:
            f.write(open(os.path.join(HERE, 'polymarket_mm.ini')).read() + '\n'
                    + open(os.path.join(HERE, 'sweep.ini')).read())
        default_ini = os.path.join(tmp, 'default.ini')
        with open(os.path.join(root, 'config', 'default.ini')) as f:
            text, n_blocks = strip_sweep_blocks(f.read())
        with open(default_ini, 'w') as f:
            f.write(text)
        assert n_blocks > 0 and '[sweep]' in text and '[sweep.' not in text
        c_path, exe = os.path.join(tmp, 'h.c'), os.path.join(tmp, 'h')
        with open(c_path, 'w') as f:
            f.write(HARNESS % FUNCS)
        subprocess.run([os.environ.get('CC', 'cc'), '-O1', '-Wall', '-Wno-unused-function',
                        '-I' + os.path.join(root, 'src'), c_path, '-o', exe], check=True)
        for extra in ([], ['--env.split=0', '--eval_env.split=1', '--env.data_dir=/x/y z/build',
                           '--train.horizon=256']):
            out = subprocess.run([exe, default_ini, cfg, *extra], check=True,
                                 capture_output=True, text=True).stdout
            print(('argv ' if extra else 'ini  ') + ' | '.join(out.strip().splitlines()))
            assert 'eval_env: final eval with [env] split = 1 (was 0)' in out
        # the C++ build compiles this too: nvcc -std=c++17 (casts, no VLAs)
        subprocess.run(['g++', '-std=c++17', '-fsyntax-only', '-Wall', '-Wno-unused-function',
                        '-x', 'c++', '-I' + os.path.join(root, 'src'), c_path], check=True)

        src = os.path.join(tmp, 'pufferl.cu')
        shutil.copy(os.path.join(root, 'src', 'pufferl.cu'), src)
        with open(src) as f:
            orig = f.read()
        once, changed = patch_source(orig)
        twice, changed2 = patch_source(once)
        assert changed and not changed2 and once == twice
        assert once.count('eval_env_apply(ini)') == 1 and once.count('eval_env_restore(ini,') == 1
        assert once.index('static char** eval_env_apply') < once.index(FUNC_ANCHOR)
        assert len(once) - len(orig) == len(FUNCS.lstrip('\n')) + 1 + len(EVAL_NEW) - len(EVAL_OLD)
        rev = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True).stdout.strip()
        print(f'pufferl.cu ({rev}): one insertion each, idempotent; '
              f'default.ini: {n_blocks} [sweep.*] blocks removed; compiles as C and C++17')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--selftest':
        sys.path.insert(0, HERE)
        selftest(args[1] if len(args) > 1 else os.environ.get(
            'PUFFERLIB_DIR', '/tmp/claude-0/-home-user-MARs/bd289cf3-b2b3-5940-9bd3-6798f7649804/'
            'scratchpad/pufferlib5'))
        print('ok')
    elif len(args) == 1:
        apply(args[0])
    else:
        sys.exit(__doc__)
