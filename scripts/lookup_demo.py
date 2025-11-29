import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument('--month', required=True)
ap.add_argument('--entity', required=True)
args = ap.parse_args()

path = os.path.join('features','online', f'online_{args.month}.json')
if not os.path.exists(path):
    print('Online file not found:', path, file=sys.stderr); sys.exit(2)

data = json.load(open(path, 'r', encoding='utf-8'))
rec = data.get(args.entity)
if not rec:
    print('Entity not found. Available keys:', list(data.keys())); sys.exit(1)

print('FEATURES for', args.entity)
for k,v in rec.items():
    print(f'  {k}: {v}')