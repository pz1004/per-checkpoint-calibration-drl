"""Check the scope of omission and arithmetic claims from existing audit records."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def run(reference, diagnostic, output):
    rules = {x['point_id']: x for x in json.loads((reference/'rules.json').read_text())}
    ids = ['E:GLR:'+f+':additional_2'+suffix for f in ('arithmetic', 'quantile90')
           for suffix in ('', ':scaling_only', ':loco:omit_808', ':scaling_only:omit_808')]
    records = {key: {} for key in ids}
    with (reference/'stream_reference.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['point_id'] in records:
                records[row['point_id']][row['stream_key']] = row
    comparisons = []
    expected_gains = ['E:GLR:seed_805_rep_1_gradual_updating_down', 'E:GLR:seed_817_rep_0_gradual_updating_down']
    for family in ('arithmetic', 'quantile90'):
        source = 'E:GLR:'+family+':additional_2'
        for rule in ('loco', 'scaling_only'):
            before = source if rule == 'loco' else source+':scaling_only'
            after = source+':'+rule+':omit_808'
            old, new = records[before], records[after]
            retained = {key for key, x in old.items() if x['checkpoint_seed'] != '808'}
            assert retained == set(new) and len(retained) == 456
            primary = {key for key, x in new.items() if x['condition'] == 'gradual_updating_down'}
            assert len(primary) == 76
            gain = sorted(k for k in primary if old[k]['binary64_fired']=='False' and new[k]['binary64_fired']=='True')
            loss = sorted(k for k in primary if old[k]['binary64_fired']=='True' and new[k]['binary64_fired']=='False')
            assert gain == (expected_gains if family=='arithmetic' else []) and not loss
            changed = sorted(k for k in retained if old[k]['binary64_fired'] != new[k]['binary64_fired'])
            stable = [k for k in changed if new[k]['role'] != 'changed']
            assert stable == ['E:GLR:seed_810_rep_5_stable_frozen']
            assert old[stable[0]]['binary64_fired']=='False' and new[stable[0]]['binary64_fired']=='True'
            stable_counts = [sum(x['tuning']=='True' and x['binary64_fired']=='True' for x in group)
                             for group in ([old[k] for k in retained], list(new.values()))]
            assert stable_counts == [1, 2]
            for a, b in ((source, source+':scaling_only'), (source+':loco:omit_808', source+':scaling_only:omit_808')):
                assert Fraction(rules[a]['exact_theta']) == Fraction(rules[b]['exact_theta'])
            comparisons.append(dict(family=family, rule=rule, primary_gained_ids=gain, primary_lost_ids=loss,
                                    changed_stable_ids=stable, retained_stable_counts=stable_counts,
                                    exact_global_levels_equal_between_rules_before_and_after=True))
    coordinate = Counter()
    with (diagnostic/'stream_decisions.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            alt = (float(row['theta'])*float(row['divisor']))*float(row['applied_correction'])
            assert alt == float(row['reparameterized_threshold'])
            differs = (float(row['maximum']) > alt) != (row['fired']=='True')
            assert differs == (row['coordinate_decision_differs']=='True')
            if differs:
                assert row['role'] != 'changed'
                coordinate['omitted' if row['omitted_checkpoint'] else 'full'] += 1
    assert dict(coordinate) == {'full': 18, 'omitted': 24}
    sources = [reference/'rules.json', reference/'stream_reference.csv', diagnostic/'stream_decisions.csv']
    hashes = {('reference/' if p.parent == reference else 'diagnostic/')+p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(status='PASS', omission_comparisons=comparisons, alternative_product_order_stable_differences=dict(coordinate),
                  input_hashes=hashes, scope='Read-only comparisons of existing fits; no new fitting, cap grid or experiment.')
    with output.open('x') as handle:
        handle.write(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.reference, args.diagnostic, args.output), indent=2))
