import argparse
import sys
sys.path.append('/home/divc_col2/Group_3/DeCoF_backup/src')
from train2 import train_decof  # type: ignore

args = argparse.Namespace(
    config="/home/divc_col2/Group_3/DeCoF_backup/src/configs/base.json",
    session_name="DeCoF"
)

def main():
    train_decof(args)

if __name__ == "__main__":
    main()