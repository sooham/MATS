"""Execute only the new analysis notebooks in isolated CPU kernels, saving each cell."""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import nbformat
from nbclient import NotebookClient


class SavingClient(NotebookClient):
    async def async_execute_cell(self, cell, cell_index, *args, **kwargs):
        if self.selected_cells is not None and cell_index not in self.selected_cells:
            return cell
        print(datetime.now(timezone.utc).isoformat(), 'cell', cell_index,
              cell.source.splitlines()[0][:110] if cell.source else '', flush=True)
        try:
            return await super().async_execute_cell(cell, cell_index, *args, **kwargs)
        finally:
            nbformat.write(self.nb, self.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--cells', help='Comma-separated cell indices to execute; retain other cells and outputs')
    args = parser.parse_args()
    path = args.notebook.resolve()
    assert path.name.startswith(('18_', '19_', '20_')), 'Only new exploration notebooks may be executed'
    notebook = nbformat.read(path, as_version=4)
    client = SavingClient(notebook, timeout=None, kernel_name='python3',
                          resources={'metadata': {'path': str(path.parent)}})
    client.output_path = path
    client.selected_cells = set(map(int, args.cells.split(','))) if args.cells else None
    client.execute()
    print('COMPLETE', path, flush=True)
