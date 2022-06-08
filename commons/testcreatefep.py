import os

def each_chunk(stream, separator):
  buffer = ''
  while True:  # until EOF
    chunk = stream.read(4096)  # I propose 4096 or so
    if not chunk:  # EOF?
      yield buffer
      break
    buffer += chunk
    while True:  # until no separator is found
      try:
        part, buffer = buffer.split(separator, 1)
      except ValueError:
        break
      else:
        yield part

base = '../data/FEPBenchmark'

for subdir in os.listdir(base):
    ligand_cntr = 0
    subpath = os.path.join(base, subdir)
    with open(os.path.join(subpath, 'all.sdf')) as f:
        for chunk in each_chunk(f, separator='$$$$'):
            if chunk.strip() != '':
              writeto = open(os.path.join(subpath, f'ligand_{ligand_cntr}.sdf'), 'w')
              writeto.write(chunk.strip())
              writeto.close
              ligand_cntr += 1
