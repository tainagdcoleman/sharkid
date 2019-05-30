import argparse
import pathlib
import os

from functools import partial
from multiprocessing import Pool 

from fin import Fin 

import matplotlib.pyplot as plt

def save_fin(fin, output):
    fin.save(output)

def fin_scores(fin, others):
    scores = [(other.path_image.stem, fin.similarity(other))
              for other in others if fin != other]

    return sorted(scores, key=lambda x: x[1])

def plot_scores(scores, output):
    fig, ax = plt.subplots(1)

    xs, ys = zip(*scores)
    ax.bar(xs, ys)

    output = pathlib.Path(output)
    if output.is_dir():
        raise FileExistsError(f'{output} is a directory.')
    output.parent.mkdir(exist_ok=True, parents=True)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()
    
    fig.savefig(str(output))
    plt.close(fig)

def main(input_directory, out):
    input_directory = pathlib.Path(input_directory)
    out = pathlib.Path(out)

    if not input_directory.is_dir():
        raise FileNotFoundError('Input directory not found')

    pool = Pool(os.cpu_count())

    print('Processing Fins... ', end='', flush=True)
    fins = pool.map(Fin, input_directory.glob('*.png'))
    print('Done!')

    fins = [fin for fin in fins if fin.is_processed]
    save_paths = [out.joinpath('keypoints', fin.path_image.name)
                  for fin in fins]

    print('Creating keypoints plots... ', end='', flush=True)
    pool.starmap(save_fin, zip(fins, save_paths))
    print('Done!')

    print('Calculating comparison scores... ', end='', flush=True)
    scores = pool.starmap(fin_scores, ((fin, fins) for fin in fins))
    print('Done!')

    score_plot_paths = [out.joinpath('scores', fin.path_image.name)
                        for fin in fins]


    print('Creating scores plots... ', end='', flush=True)
    pool.starmap(plot_scores, zip(scores, score_plot_paths))
    print('Done!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_directory',
    )
    parser.add_argument(
        '--out',
        default='out',
        help='output directory',
    )

    args = parser.parse_args()
    
    main(args.input_directory, args.out)