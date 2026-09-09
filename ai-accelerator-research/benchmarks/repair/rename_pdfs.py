import os

files = [
    'synthesis_passrate.pdf',
    'transfer_speedup.pdf',
    'scalability.pdf',
    'cold_start_convergence.pdf',
    'ablation_repair.pdf',
    'autompg_importance.pdf'
]

for f in files:
    if os.path.exists(f):
        new_name = f.replace('.pdf', '_v2.pdf')
        os.rename(f, new_name)
        print(f'Renamed {f} -> {new_name}')