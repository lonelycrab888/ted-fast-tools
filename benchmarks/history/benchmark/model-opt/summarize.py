"""Assemble verified TED model benchmarks from the stored per-run measurements."""
import json
import os
from pathlib import Path
import statistics

from validate import raw_rows


ROOT = Path(os.environ.get("TED_BENCHMARK_DIR", Path(__file__).resolve().parent))
SOURCE_VARIANT = os.environ.get("TED_BENCHMARK_VARIANT", "ted-fast")


def read(path):
    return json.loads(path.read_text())


def series(pattern, method=None):
    paths = [ROOT/pattern.format(i=i) for i in (1, 2)]
    times = []
    for path in paths:
        timing = read(path/'timing.json')
        assert timing['inputs'] == 128 and not timing.get('profile_enabled', False)
        times.append(timing['wall_seconds'])
        if method:
            reference = ROOT/'ted-v2-1'
            assert raw_rows(path, method) == raw_rows(reference, method)
            filename = f'chopping_{method}.txt'
            assert sorted((path/filename).read_text().splitlines()) == sorted((reference/filename).read_text().splitlines())
        else:
            assert (timing['overlap_gpu_models'] and timing['unidoc_backend'] == 'fast'
                    and timing['unidoc_batch_size'] == 32 and list(timing['gpus']) == ['4'])
    return {'directories': [p.name for p in paths], 'wall_seconds': times, 'mean_seconds': statistics.mean(times)}


def summarize():
    points = read(ROOT/'ipa-kernels.json')
    chain = read(ROOT/'chainsaw-features.json')
    assert all(r['distance']['equal'] and r['values']['equal'] for r in points)
    assert chain['exact'] == chain['inputs'] == 128 and all(chain['fallbacks'].values())
    pipeline_name = SOURCE_VARIANT if SOURCE_VARIANT.startswith('ted-') else f'ted-{SOURCE_VARIANT}'
    sets = {
        'merizo_gpu5_baseline': series('merizo-gpu5-v2-{i}', 'merizo'),
        'merizo_gpu5_ipa': series('merizo-gpu5-v4-{i}', 'merizo'),
        'merizo_gpu5_ipa_gru': series(f'merizo-gpu5-{SOURCE_VARIANT}-{{i}}', 'merizo'),
        'merizo_gpu7_ipa': series('merizo-gpu7-v4-{i}', 'merizo'),
        'merizo_gpu7_ipa_gru': series(f'merizo-gpu7-{SOURCE_VARIANT}-{{i}}', 'merizo'),
        'chainsaw_gpu4_baseline': series('isolated-chainsaw-v2-{i}', 'chainsaw'),
        'chainsaw_gpu4_sync_features': series('isolated-chainsaw-v4-{i}', 'chainsaw'),
        'chainsaw_gpu4_prefetch': series('chainsaw-stride-v4-{i}', 'chainsaw'),
        'ted_gpu4_baseline': series('ted-v2-{i}'),
        'ted_gpu4_v4': series('ted-v4-{i}'),
        'ted_fast_gpu4': series(f'{pipeline_name}-{{i}}'),
    }
    comparisons = {}
    for name, old, new in (
        ('merizo','merizo_gpu5_baseline','merizo_gpu5_ipa_gru'),
        ('merizo_ipa','merizo_gpu5_baseline','merizo_gpu5_ipa'),
        ('merizo_graphs','merizo_gpu7_ipa','merizo_gpu7_ipa_gru'),
        ('chainsaw','chainsaw_gpu4_baseline','chainsaw_gpu4_prefetch'),
        ('ted_v4','ted_gpu4_baseline','ted_gpu4_v4'),
        ('ted_fast','ted_gpu4_baseline','ted_fast_gpu4')):
        a, b = sets[old]['mean_seconds'], sets[new]['mean_seconds']
        comparisons[name] = {'saved_seconds': a-b, 'speedup': a/b, 'percent_reduction': 100*(1-b/a)}
    holdouts = [read(ROOT/f'holdout-{v}-{i}.json') for v in ('v2','v4','gru') for i in (1, 2)]
    full = read(ROOT/'full-validation.json')
    for report in holdouts+[full]:
        n = report['structures']
        assert all(v['rows']==v['identical']==n for v in report['raw'].values())
        assert all(v['rows']==v['rows_identical']==n for v in report['filtered_and_consensus'].values())
    assert full['structures'] == 2349
    ft = read(ROOT/f'full-{SOURCE_VARIANT}/timing.json')
    assert ft['variant'] == SOURCE_VARIANT
    return {
        'dataset': 'benchmark/holdout', 'count': 128,
        'timing_scope': 'Fresh model processes to filtered method or consensus output, exclusive of domain PDB export.',
        'runs': sets, 'comparisons': comparisons,
        'profiles': {label: read(ROOT/f'merizo-profile-{source}.json')
                     for label, source in [('v2', 'v2'), ('v4', 'v4'), ('ted-fast', SOURCE_VARIANT)]},
        'chainsaw_gpu_phases_and_layout': read(ROOT/'chainsaw-layout.json'),
        'gru_graph_microbench': [read(ROOT/f'gru-graph-{n}.json') for n in (128,512,1400)],
        'cuda_kernel_checks': len(points), 'chainsaw_feature_check': chain,
        'full_validation': {'structures': full['structures'], 'raw_identical': {k:v['identical'] for k,v in full['raw'].items()},
            'filtered_identical': {k:v['rows_identical'] for k,v in full['filtered_and_consensus'].items()},
            'high': full['high'], 'medium': full['medium'], 'no_retained_domains': len(full['without_high_medium']),
            'wall_seconds': ft['wall_seconds'], 'gpus': list(ft['gpus']), 'variant': ft['variant']}
    }


if __name__ == '__main__':
    summary = summarize()
    Path(__file__).with_name('results.json').write_text(json.dumps(summary, indent=2)+'\n')
    for k,v in summary['runs'].items():
        print(k, '/'.join(f'{x:.2f}' for x in v['wall_seconds']), f"mean {v['mean_seconds']:.3f}")
    for k,v in summary['comparisons'].items():
        print(k, round(v['speedup'],3), round(v['percent_reduction'],2))
    print(summary['full_validation'])
