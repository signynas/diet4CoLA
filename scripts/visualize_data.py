import argparse

from diet4cola.utils import load_cortex, plot_2d_array
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Cortex Example Generator"')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to cortex file to show')
    args = parser.parse_args()
    file_path = Path(args.file)
    if not file_path.exists:
        raise ValueError(f'File {file_path} does not exist!')
    cortex_data = load_cortex(str(file_path))
    plot_2d_array(cortex_data)

if __name__ == "__main__":
    main()