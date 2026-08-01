import pandas as pd
import numpy as np

df = pd.read_csv('benchmarks/results.csv')

def format_evs(row):
    evs = row['events_per_s']
    t = row['wall_time_s']
    t_std = row['wall_time_std']
    evs_std = evs * (t_std / t) if t > 0 and t_std > 0 else 0.0

    if evs == 0 or np.isnan(evs):
        return '---'
    p = int(np.floor(np.log10(evs)))
    val = evs / (10**p)
    std_val = evs_std / (10**p)

    return f"$({val:.2f} \\pm {std_val:.2f})\\times10^{{{p}}}$"

print('=== MARKETS SWEEP ===')
for m in [64, 256, 1024, 4096, 16384]:
    sub = df[(df.sweep == 'markets') & (df.n_markets == m)]
    if sub.empty:
        continue
    row_cpu = sub[sub.backend == 'cpu_numpy'].iloc[0] if not sub[sub.backend == 'cpu_numpy'].empty else None
    row_torch = sub[sub.backend == 'torch_gpu'].iloc[0] if not sub[sub.backend == 'torch_gpu'].empty else None
    row_jax = sub[sub.backend == 'jax_gpu'].iloc[0] if not sub[sub.backend == 'jax_gpu'].empty else None
    row_naive = sub[sub.backend == 'naive_cuda'].iloc[0] if not sub[sub.backend == 'naive_cuda'].empty else None
    row_ks = sub[sub.backend == 'kineticsim_cuda'].iloc[0] if not sub[sub.backend == 'kineticsim_cuda'].empty else None

    str_cpu = format_evs(row_cpu) if row_cpu is not None else '---'
    str_torch = format_evs(row_torch) if row_torch is not None else '---'
    str_jax = format_evs(row_jax) if row_jax is not None else '---'
    str_naive = format_evs(row_naive) if row_naive is not None else '---'
    str_ks = format_evs(row_ks) if row_ks is not None else '---'

    vs_cpu = f'{row_ks["speedup_vs_cpu"]:.0f}\\times' if row_ks is not None and not np.isnan(row_ks["speedup_vs_cpu"]) else '---'
    vs_torch = f'{row_ks["speedup_vs_torch"]:.0f}\\times' if row_ks is not None and not np.isnan(row_ks["speedup_vs_torch"]) else '---'
    vs_jax = f'{row_ks["speedup_vs_jax"]:.0f}\\times' if row_ks is not None and not np.isnan(row_ks["speedup_vs_jax"]) else '---'

    print(f'M={m:<5d} & {str_cpu} & {str_torch} & {str_jax} & {str_naive} & \\mathbf{{{str_ks}}} & {vs_cpu} & {vs_torch} & {vs_jax} \\\\')

print('\n=== AGENTS SWEEP ===')
for a in [16, 64, 256, 1024]:
    sub = df[(df.sweep == 'agents') & (df.n_agents == a)]
    if sub.empty:
        continue
    row_cpu = sub[sub.backend == 'cpu_numpy'].iloc[0] if not sub[sub.backend == 'cpu_numpy'].empty else None
    row_torch = sub[sub.backend == 'torch_gpu'].iloc[0] if not sub[sub.backend == 'torch_gpu'].empty else None
    row_jax = sub[sub.backend == 'jax_gpu'].iloc[0] if not sub[sub.backend == 'jax_gpu'].empty else None
    row_naive = sub[sub.backend == 'naive_cuda'].iloc[0] if not sub[sub.backend == 'naive_cuda'].empty else None
    row_ks = sub[sub.backend == 'kineticsim_cuda'].iloc[0] if not sub[sub.backend == 'kineticsim_cuda'].empty else None

    str_cpu = format_evs(row_cpu) if row_cpu is not None else '---'
    str_torch = format_evs(row_torch) if row_torch is not None else '---'
    str_jax = format_evs(row_jax) if row_jax is not None else '---'
    str_naive = format_evs(row_naive) if row_naive is not None else '---'
    str_ks = format_evs(row_ks) if row_ks is not None else '---'

    vs_cpu = f'{row_ks["speedup_vs_cpu"]:.0f}\\times' if row_ks is not None and not np.isnan(row_ks["speedup_vs_cpu"]) else '---'
    vs_torch = f'{row_ks["speedup_vs_torch"]:.0f}\\times' if row_ks is not None and not np.isnan(row_ks["speedup_vs_torch"]) else '---'
    vs_jax = f'{row_ks["speedup_vs_jax"]:.0f}\\times' if row_ks is not None and not np.isnan(row_ks["speedup_vs_jax"]) else '---'

    print(f'A={a:<5d} & {str_cpu} & {str_torch} & {str_jax} & {str_naive} & \\mathbf{{{str_ks}}} & {vs_cpu} & {vs_torch} & {vs_jax} \\\\')

print('\n=== FIXED WORKLOAD ===')
sub = df[df.sweep == 'fixed']
for b in ['cpu_numpy', 'torch_gpu', 'torch_graph', 'torch_compile', 'jax_gpu', 'naive_cuda', 'kineticsim_cuda']:
    row = sub[sub.backend == b]
    if row.empty:
        continue
    row = row.iloc[0]
    evs = row['events_per_s']
    t = row['wall_time_s']
    t_std = row['wall_time_std']
    evs_std = evs * (t_std / t) if t > 0 and t_std > 0 else 0.0

    p = int(np.floor(np.log10(evs)))
    str_evs = f"$({evs/(10**p):.3f} \\pm {evs_std/(10**p):.3f}) \\times 10^{p}$"

    str_time = f"${t*1000.0:.1f} \\pm {t_std*1000.0:.1f}$"
    ns_event = row['ns_per_event']

    print(f"{row['backend']:15s} & {str_evs} & {str_time} & {ns_event:.3f} \\\\")
